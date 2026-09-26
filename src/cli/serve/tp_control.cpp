#include "src/cli/serve/tp_control.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>

#include "src/core/crypto/sha256.hpp"

namespace gufo::server {
namespace {

constexpr std::uint32_t kMagic = 0x54504331U;  // "TPC1"
constexpr std::uint16_t kVersion = 7;          // C1/C2 envelopes plus rank-1
                                               // executor instructions
constexpr std::uint16_t kHello = 1;
constexpr std::uint16_t kCommand = 2;
constexpr std::uint16_t kResponse = 3;
constexpr std::size_t kMaxPayloadBytes = 16U << 20;
constexpr std::size_t kMaxPromptTokens = 1U << 20;
constexpr std::size_t kMaxCohortMembers = 2;
constexpr std::size_t kMaxClientIdBytes = 4096;
constexpr std::size_t kMaxErrorBytes = 1U << 20;
constexpr std::size_t kMaxAuthTokenBytes = 4096;
constexpr auto kIoTimeout = std::chrono::seconds(30);

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

std::string SystemError(const char* operation) {
  return std::string(operation) + ": " + std::strerror(errno);
}

/// Bounds blocking sends and pre-handshake receives, and turns on TCP
/// keepalive so a peer whose host died is reported even on an idle channel.
void SetStartupSocketOptions(int fd, std::chrono::milliseconds io_timeout) {
  const auto milliseconds = std::max<std::int64_t>(io_timeout.count(), 1);
  timeval timeout{};
  timeout.tv_sec = static_cast<time_t>(milliseconds / 1000);
  timeout.tv_usec = static_cast<suseconds_t>((milliseconds % 1000) * 1000);
  (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  const int enabled = 1;
  (void)::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &enabled, sizeof(enabled));
#if defined(TCP_KEEPIDLE) && defined(TCP_KEEPINTVL) && defined(TCP_KEEPCNT)
  // Probe after one idle minute, then every ten seconds: a dead host is
  // reported about two minutes after it stops answering.
  const int idle_seconds = 60;
  const int interval_seconds = 10;
  const int probes = 6;
  (void)::setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle_seconds,
                     sizeof(idle_seconds));
  (void)::setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &interval_seconds,
                     sizeof(interval_seconds));
  (void)::setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &probes, sizeof(probes));
#endif
}

/// After the handshake a rank waits for the next command or response for as
/// long as the server stays idle, so a receive timeout would end the pair.
void ClearReceiveTimeout(int fd) {
  const timeval none{};
  (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &none, sizeof(none));
}

void AppendU32(std::vector<std::uint8_t>* out, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8) {
    out->push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void AppendU64(std::vector<std::uint8_t>* out, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8) {
    out->push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

bool ReadU32(std::span<const std::uint8_t> data, std::size_t* offset,
             std::uint32_t* value, std::string* error) {
  if (*offset > data.size() || data.size() - *offset < sizeof(std::uint32_t)) {
    SetError(error, "TP control payload is truncated");
    return false;
  }
  std::uint32_t result = 0;
  for (unsigned shift = 0; shift < 32; shift += 8) {
    result |= static_cast<std::uint32_t>(data[(*offset)++]) << shift;
  }
  *value = result;
  return true;
}

bool ReadU64(std::span<const std::uint8_t> data, std::size_t* offset,
             std::uint64_t* value, std::string* error) {
  if (*offset > data.size() || data.size() - *offset < sizeof(std::uint64_t)) {
    SetError(error, "TP control payload is truncated");
    return false;
  }
  std::uint64_t result = 0;
  for (unsigned shift = 0; shift < 64; shift += 8) {
    result |= static_cast<std::uint64_t>(data[(*offset)++]) << shift;
  }
  *value = result;
  return true;
}

void AppendBytes(std::vector<std::uint8_t>* out,
                 std::span<const std::uint8_t> bytes) {
  out->insert(out->end(), bytes.begin(), bytes.end());
}

bool ReadBytes(std::span<const std::uint8_t> data, std::size_t* offset,
               std::span<std::uint8_t> bytes, std::string* error) {
  if (*offset > data.size() || bytes.size() > data.size() - *offset) {
    SetError(error, "TP control payload is truncated");
    return false;
  }
  std::memcpy(bytes.data(), data.data() + *offset, bytes.size());
  *offset += bytes.size();
  return true;
}

[[nodiscard]] bool IsZeroDigest(const TpPlanDigest& digest) noexcept {
  return std::ranges::all_of(digest,
                             [](std::uint8_t byte) { return byte == 0; });
}

[[nodiscard]] bool HasLegacyCommandFields(const TpControlCommand& command) {
  return command.max_tokens != 0 || command.cache_prompt ||
         command.cache_prefix_tokens != 0 || !command.prompt_tokens.empty() ||
         !command.client_id.empty() || command.sampled;
}

/// An instruction names one model call and its arguments; every argument the
/// call does not take must be zero, so two encodings of one call cannot differ.
[[nodiscard]] bool ValidateInstruction(std::uint64_t sequence,
                                       const TpInstruction& instruction,
                                       std::string* error) {
  const auto only = [&](bool token, bool offset, bool count, bool prompt_size,
                        bool digest) {
    return (token || instruction.token == 0) &&
           (offset || instruction.offset == 0) &&
           (count || instruction.count == 0) &&
           (prompt_size || instruction.prompt_size == 0) &&
           (digest || instruction.digest == 0);
  };
  bool valid = false;
  switch (instruction.op) {
    case TpInstructionOp::kInvalidate:
      valid = only(false, false, false, false, false);
      break;
    case TpInstructionOp::kPrefill:
      valid = only(false, true, true, true, false) && instruction.count != 0 &&
              instruction.prompt_size > instruction.offset &&
              instruction.prompt_size <= kMaxPromptTokens;
      break;
    case TpInstructionOp::kAdvance:
      valid = only(true, false, false, false, false) && instruction.token >= 0;
      break;
    case TpInstructionOp::kDecode:
      valid = only(false, false, true, false, false) && instruction.count > 1;
      break;
    case TpInstructionOp::kEnd:
      valid = only(false, false, true, false, true);
      break;
    case TpInstructionOp::kNone:
      break;
  }
  if (!valid) {
    SetError(error, "TP instruction has invalid arguments for its operation");
    return false;
  }
  // Only a reset can happen between requests: the continuation cache makes it
  // on its own schedule. Every other call belongs to a request.
  if (sequence == 0 && instruction.op != TpInstructionOp::kInvalidate) {
    SetError(error, "TP instruction outside a request must be a reset");
    return false;
  }
  return true;
}

[[nodiscard]] bool HasLegacyResponseFields(const TpControlResponse& response) {
  return !response.tokens.empty() || response.draft_tokens != 0 ||
         response.draft_accepted_tokens != 0 ||
         response.cached_prompt_tokens != 0 ||
         response.cache_snapshot_bytes != 0;
}

[[nodiscard]] std::vector<TpControlMemberRequest> EffectiveCommandMembers(
    const TpControlCommand& command) {
  if (command.kind == TpControlCommandKind::kSingle &&
      !command.members.empty()) {
    return command.members;
  }
  if (command.kind == TpControlCommandKind::kSingle) {
    return {{
        .member_id = command.sequence,
        .max_tokens = command.max_tokens,
        .cache_prompt = command.cache_prompt,
        .cache_prefix_tokens = command.cache_prefix_tokens,
        .prompt_tokens = command.prompt_tokens,
        .client_id = command.client_id,
    }};
  }
  return command.members;
}

void HashString(crypto::Sha256Hasher* hash, std::string_view value) {
  std::vector<std::uint8_t> encoded;
  AppendU32(&encoded, static_cast<std::uint32_t>(value.size()));
  encoded.insert(encoded.end(), value.begin(), value.end());
  hash->Update(encoded);
}

void HashU32(crypto::Sha256Hasher* hash, std::uint32_t value) {
  std::array<std::uint8_t, 4> encoded{};
  for (unsigned shift = 0; shift < 32; shift += 8) {
    encoded[shift / 8] = static_cast<std::uint8_t>(value >> shift);
  }
  hash->Update(encoded);
}

void HashU64(crypto::Sha256Hasher* hash, std::uint64_t value) {
  std::array<std::uint8_t, 8> encoded{};
  for (unsigned shift = 0; shift < 64; shift += 8) {
    encoded[shift / 8] = static_cast<std::uint8_t>(value >> shift);
  }
  hash->Update(encoded);
}

void HashPromptTokens(crypto::Sha256Hasher* hash,
                      std::span<const std::int32_t> tokens) {
  std::vector<std::uint8_t> encoded;
  encoded.reserve(tokens.size() * sizeof(std::uint32_t));
  for (const auto token : tokens) {
    AppendU32(&encoded, static_cast<std::uint32_t>(token));
  }
  hash->Update(encoded);
}

[[nodiscard]] bool ValidateMemberRequest(const TpControlMemberRequest& member,
                                         std::string* error) {
  if (member.max_tokens == 0 || member.prompt_tokens.empty() ||
      member.prompt_tokens.size() > kMaxPromptTokens ||
      member.cache_prefix_tokens > member.prompt_tokens.size() ||
      member.client_id.size() > kMaxClientIdBytes ||
      std::ranges::any_of(member.prompt_tokens,
                          [](std::int32_t token) { return token < 0; })) {
    SetError(error, "TP control cohort member request is invalid");
    return false;
  }
  return true;
}

[[nodiscard]] bool ValidateMemberResponse(const TpControlMemberResponse& member,
                                          std::string* error) {
  if (member.tokens.size() > kMaxPromptTokens) {
    SetError(error, "TP control cohort member response is too large");
    return false;
  }
  return true;
}

}  // namespace

TpPlanDigest ComputeTpExecutionPlanDigest(const TpControlCommand& command) {
  const auto members = EffectiveCommandMembers(command);
  crypto::Sha256Hasher hash;
  HashString(&hash, "gufo.tp-control.execution-plan.v1");
  HashU32(&hash, static_cast<std::uint32_t>(command.kind));
  HashU64(&hash, command.kind == TpControlCommandKind::kSingle
                     ? command.sequence
                     : command.cohort_id);
  HashU32(&hash, static_cast<std::uint32_t>(members.size()));
  HashU32(&hash, static_cast<std::uint32_t>(members.size()));
  // Execution shape. Every plan that exists today is serial: one request
  // decodes at a time and no prefill is batched. A C2 cohort is two sequential
  // single-row executions, not one two-row one, so a cohort is NOT a width-2
  // execution.
  //
  // What each position MEANS was never established, so the values are left
  // exactly as they were rather than being given invented names. Do not
  // reorder or reinterpret them: that would silently change every C2 digest.
  //
  // Known limitation: this function cannot distinguish a serial cohort from a
  // batched one, because the execution width is not carried on the wire. Both
  // ranks hash the same command bytes, so comparing digests proves the command
  // was not mutated -- it does NOT prove the two ranks will execute the same
  // way. Closing that needs either an execution-width field on the command or
  // a distinct command kind, and neither should be guessed before a batched
  // runner exists to describe.
  if (command.kind == TpControlCommandKind::kCohort2Ar) {
    constexpr std::uint32_t kC2Shape0 = 0;
    constexpr std::uint32_t kC2Shape1 = 1;
    constexpr std::uint32_t kC2Shape2 = 0;
    constexpr std::uint32_t kC2Shape3 = 1;
    HashU32(&hash, kC2Shape0);
    HashU32(&hash, kC2Shape1);
    HashU32(&hash, kC2Shape2);
    HashU32(&hash, kC2Shape3);
  } else {
    // The C1 layout is a single field, not a prefix of the C2 layout, and
    // nothing establishes that it denotes the same quantity as any of the
    // four. Hashed as zero, unchanged.
    constexpr std::uint32_t kC1Shape = 0;
    HashU32(&hash, kC1Shape);
  }
  for (const auto& member : members) {
    HashU64(&hash, member.member_id);
    HashU32(&hash, member.max_tokens);
    HashU32(&hash, static_cast<std::uint32_t>(member.prompt_tokens.size()));
    HashPromptTokens(&hash, member.prompt_tokens);
  }
  return hash.Finish();
}

TpPlanDigest ComputeTpCachePlanDigest(const TpControlCommand& command) {
  const auto members = EffectiveCommandMembers(command);
  crypto::Sha256Hasher hash;
  HashString(&hash, "gufo.tp-control.cache-plan.v1");
  HashU32(&hash, static_cast<std::uint32_t>(command.kind));
  HashU64(&hash, command.kind == TpControlCommandKind::kSingle
                     ? command.sequence
                     : command.cohort_id);
  HashU32(&hash, static_cast<std::uint32_t>(members.size()));
  for (const auto& member : members) {
    HashU64(&hash, member.member_id);
    HashU32(&hash, member.cache_prompt ? 1U : 0U);
    HashU32(&hash, member.cache_prefix_tokens);
  }
  return hash.Finish();
}

bool ValidateTpControlCommand(const TpControlCommand& command,
                              std::string* error) {
  if (command.kind != TpControlCommandKind::kInstruction &&
      command.instruction != TpInstruction{}) {
    SetError(error, "TP request command carries an instruction");
    return false;
  }
  if (command.kind == TpControlCommandKind::kSingle) {
    if (command.members.size() > 1 ||
        (!command.members.empty() && HasLegacyCommandFields(command))) {
      SetError(error, "TP C1 command has an invalid member envelope");
      return false;
    }
    const auto members = EffectiveCommandMembers(command);
    if (members.size() != 1 || members.front().member_id != command.sequence) {
      SetError(error, "TP C1 command must contain its sequence-scoped member");
      return false;
    }
    return ValidateMemberRequest(members.front(), error);
  }
  if (command.kind == TpControlCommandKind::kInstruction) {
    // An instruction is one model call within a request that a previous
    // kSingle already described. Anything else here would mean the two
    // messages disagree about the request, so every request field must be
    // empty: no members, no prompt, no cache request, no plan digest, and no
    // cohort scope of its own.
    const TpPlanDigest zero_digest{};
    if (!command.members.empty() || HasLegacyCommandFields(command) ||
        command.cohort_id != 0 ||
        command.execution_plan_digest != zero_digest ||
        command.cache_plan_digest != zero_digest) {
      SetError(error, "TP instruction carries request fields");
      return false;
    }
    return ValidateInstruction(command.sequence, command.instruction, error);
  }
  if (command.kind != TpControlCommandKind::kCohort2Ar) {
    SetError(error, "TP control command kind is invalid");
    return false;
  }
  if (command.cohort_id == 0 || HasLegacyCommandFields(command) ||
      command.members.size() != kMaxCohortMembers) {
    SetError(error, "TP C2 command envelope is invalid");
    return false;
  }
  for (std::size_t index = 0; index < command.members.size(); ++index) {
    const auto& member = command.members[index];
    if (!ValidateMemberRequest(member, error)) {
      return false;
    }
    if (member.member_id == 0 || member.cache_prompt ||
        member.cache_prefix_tokens != 0) {
      SetError(error, "TP C2 requires uncached AR members with nonzero IDs");
      return false;
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (command.members[previous].member_id == member.member_id) {
        SetError(error, "TP C2 member IDs must be unique");
        return false;
      }
    }
  }
  if (command.execution_plan_digest != ComputeTpExecutionPlanDigest(command) ||
      command.cache_plan_digest != ComputeTpCachePlanDigest(command)) {
    SetError(error, "TP C2 plan digest mismatch");
    return false;
  }
  return true;
}

bool ValidateTpControlResponse(const TpControlResponse& response,
                               std::string* error) {
  if (response.error.size() > kMaxErrorBytes) {
    SetError(error, "TP control response error is too large");
    return false;
  }
  if (response.kind == TpControlResponseKind::kSingle) {
    if (response.members.size() > 1 ||
        (!response.members.empty() && HasLegacyResponseFields(response))) {
      SetError(error, "TP C1 response has an invalid member envelope");
      return false;
    }
    if (response.members.empty()) {
      if (response.tokens.size() > kMaxPromptTokens ||
          response.cached_prompt_tokens > kMaxPromptTokens) {
        SetError(error, "TP C1 response is too large");
        return false;
      }
      return true;
    }
    if (response.members.front().member_id != response.sequence ||
        response.members.front().cached_prompt_tokens > kMaxPromptTokens ||
        !ValidateMemberResponse(response.members.front(), error)) {
      if (error != nullptr && error->empty()) {
        SetError(error, "TP C1 response has an invalid sequence-scoped member");
      }
      return false;
    }
    return true;
  }
  if (response.kind != TpControlResponseKind::kCohort2Ar) {
    SetError(error, "TP control response kind is invalid");
    return false;
  }
  if (response.cohort_id == 0 || HasLegacyResponseFields(response) ||
      response.members.size() != kMaxCohortMembers ||
      IsZeroDigest(response.execution_plan_digest) ||
      IsZeroDigest(response.cache_plan_digest)) {
    SetError(error, "TP C2 response envelope is invalid");
    return false;
  }
  for (std::size_t index = 0; index < response.members.size(); ++index) {
    const auto& member = response.members[index];
    if (!ValidateMemberResponse(member, error)) {
      return false;
    }
    if (member.member_id == 0 || member.draft_tokens != 0 ||
        member.draft_accepted_tokens != 0 || member.cached_prompt_tokens != 0 ||
        member.cache_snapshot_bytes != 0) {
      SetError(error, "TP C2 response requires uncached AR member results");
      return false;
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (response.members[previous].member_id == member.member_id) {
        SetError(error, "TP C2 response member IDs must be unique");
        return false;
      }
    }
  }
  return true;
}

TpControlChannel::TpControlChannel(int fd, std::uint32_t rank,
                                   std::uint16_t port)
    : fd_(fd), port_(port), rank_(rank) {
  if (port_ == 0) {
    sockaddr_in address{};
    socklen_t length = sizeof(address);
    if (::getsockname(fd_, reinterpret_cast<sockaddr*>(&address), &length) ==
        0) {
      port_ = ntohs(address.sin_port);
    }
  }
}

std::shared_ptr<TpControlChannel> TpControlChannel::Listen(
    std::uint16_t port, std::string* error,
    std::chrono::milliseconds io_timeout) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    SetError(error, SystemError("TP control socket"));
    return nullptr;
  }
  int reuse = 1;
  (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(port);
  if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
      ::listen(fd, 1) != 0) {
    SetError(error, SystemError("TP control bind/listen"));
    ::close(fd);
    return nullptr;
  }
  pollfd descriptor{.fd = fd, .events = POLLIN, .revents = 0};
  const int poll_result = ::poll(&descriptor, 1, 30000);
  if (poll_result <= 0) {
    SetError(error, poll_result == 0 ? "TP control accept timed out"
                                     : SystemError("TP control poll"));
    ::close(fd);
    return nullptr;
  }
  int peer = -1;
  do {
    peer = ::accept(fd, nullptr, nullptr);
  } while (peer < 0 && errno == EINTR);
  ::close(fd);
  if (peer < 0) {
    SetError(error, SystemError("TP control accept"));
    return nullptr;
  }
  SetStartupSocketOptions(peer, io_timeout);
  return std::shared_ptr<TpControlChannel>(new TpControlChannel(peer, 0, port));
}

std::shared_ptr<TpControlChannel> TpControlChannel::Connect(
    const std::string& host, std::uint16_t port, std::string* error,
    std::chrono::milliseconds io_timeout) {
  if (host.empty()) {
    SetError(error, "TP control rank one requires a host");
    return nullptr;
  }
  const auto deadline = std::chrono::steady_clock::now() + kIoTimeout;
  for (;;) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    const std::string service = std::to_string(port);
    addrinfo* addresses = nullptr;
    const int resolve =
        ::getaddrinfo(host.c_str(), service.c_str(), &hints, &addresses);
    if (resolve != 0) {
      SetError(error, "TP control getaddrinfo: " +
                          std::string(::gai_strerror(resolve)));
      return nullptr;
    }
    int fd = -1;
    for (auto* address = addresses; address != nullptr;
         address = address->ai_next) {
      fd = ::socket(address->ai_family, address->ai_socktype,
                    address->ai_protocol);
      if (fd < 0) {
        continue;
      }
      const int flags = ::fcntl(fd, F_GETFL, 0);
      if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
        ::close(fd);
        fd = -1;
        continue;
      }
      bool connected =
          ::connect(fd, address->ai_addr, address->ai_addrlen) == 0;
      if (!connected && errno == EINPROGRESS) {
        pollfd descriptor{.fd = fd, .events = POLLOUT, .revents = 0};
        const int poll_result = ::poll(&descriptor, 1, 1000);
        int socket_error = 0;
        socklen_t error_size = sizeof(socket_error);
        connected = poll_result > 0 &&
                    ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error,
                                 &error_size) == 0 &&
                    socket_error == 0;
      }
      if (connected && ::fcntl(fd, F_SETFL, flags) == 0) {
        break;
      }
      ::close(fd);
      fd = -1;
    }
    ::freeaddrinfo(addresses);
    if (fd >= 0) {
      SetStartupSocketOptions(fd, io_timeout);
      return std::shared_ptr<TpControlChannel>(
          new TpControlChannel(fd, 1, port));
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      SetError(error, "TP control connect timed out");
      return nullptr;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

TpControlChannel::~TpControlChannel() {
  interrupted_.store(true, std::memory_order_release);
  if (fd_ >= 0) {
    ::shutdown(fd_, SHUT_RDWR);
    ::close(fd_);
    fd_ = -1;
  }
}

void TpControlChannel::Interrupt() noexcept {
  if (!interrupted_.exchange(true, std::memory_order_acq_rel) && fd_ >= 0) {
    ::shutdown(fd_, SHUT_RDWR);
  }
}

bool TpControlChannel::SendAll(const void* data, std::size_t bytes,
                               std::string* error) {
  const auto* cursor = static_cast<const std::uint8_t*>(data);
  std::size_t sent = 0;
  while (sent < bytes) {
    const ssize_t result =
        ::send(fd_, cursor + sent, bytes - sent, MSG_NOSIGNAL);
    if (result > 0) {
      sent += static_cast<std::size_t>(result);
      continue;
    }
    if (result < 0 && errno == EINTR) {
      continue;
    }
    SetError(error, SystemError("TP control send"));
    return false;
  }
  return true;
}

bool TpControlChannel::RecvAll(void* data, std::size_t bytes,
                               std::string* error) {
  auto* cursor = static_cast<std::uint8_t*>(data);
  std::size_t received = 0;
  while (received < bytes) {
    const ssize_t result = ::recv(fd_, cursor + received, bytes - received, 0);
    if (result > 0) {
      received += static_cast<std::size_t>(result);
      continue;
    }
    if (result == 0) {
      SetError(error, "TP control peer closed the channel");
      return false;
    }
    if (errno == EINTR) {
      continue;
    }
    SetError(error, SystemError("TP control receive"));
    return false;
  }
  return true;
}

bool TpControlChannel::SendFrame(std::uint16_t type, std::uint64_t sequence,
                                 const std::vector<std::uint8_t>& payload,
                                 std::string* error) {
  const std::lock_guard<std::mutex> lock(send_mutex_);
  if (interrupted_.load(std::memory_order_acquire)) {
    SetError(error, "TP control channel is interrupted");
    return false;
  }
  if (fd_ < 0 || payload.size() > kMaxPayloadBytes ||
      payload.size() > std::numeric_limits<std::uint32_t>::max()) {
    SetError(error, "TP control frame is invalid or too large");
    return false;
  }
  std::vector<std::uint8_t> header;
  header.reserve(20);
  AppendU32(&header, kMagic);
  header.push_back(static_cast<std::uint8_t>(kVersion));
  header.push_back(static_cast<std::uint8_t>(kVersion >> 8));
  header.push_back(static_cast<std::uint8_t>(type));
  header.push_back(static_cast<std::uint8_t>(type >> 8));
  AppendU32(&header, static_cast<std::uint32_t>(payload.size()));
  AppendU64(&header, sequence);
  return SendAll(header.data(), header.size(), error) &&
         (payload.empty() || SendAll(payload.data(), payload.size(), error));
}

bool TpControlChannel::ReceiveFrame(std::uint16_t type, std::uint64_t* sequence,
                                    std::vector<std::uint8_t>* payload,
                                    std::string* error) {
  const std::lock_guard<std::mutex> lock(receive_mutex_);
  if (interrupted_.load(std::memory_order_acquire)) {
    SetError(error, "TP control channel is interrupted");
    return false;
  }
  if (fd_ < 0 || sequence == nullptr || payload == nullptr) {
    SetError(error, "TP control receive state is invalid");
    return false;
  }
  std::array<std::uint8_t, 20> header{};
  if (!RecvAll(header.data(), header.size(), error)) {
    return false;
  }
  std::size_t offset = 0;
  std::uint32_t magic = 0;
  if (!ReadU32(header, &offset, &magic, error) || magic != kMagic ||
      header[4] != kVersion || header[5] != 0 ||
      header[6] != static_cast<std::uint8_t>(type) ||
      header[7] != static_cast<std::uint8_t>(type >> 8)) {
    SetError(error, "TP control frame header is invalid");
    return false;
  }
  offset += 4;  // version and frame type
  std::uint32_t bytes = 0;
  if (!ReadU32(header, &offset, &bytes, error) || bytes > kMaxPayloadBytes ||
      !ReadU64(header, &offset, sequence, error)) {
    SetError(error, "TP control frame length is invalid");
    return false;
  }
  std::vector<std::uint8_t> received(bytes);
  if (bytes != 0 && !RecvAll(received.data(), received.size(), error)) {
    return false;
  }
  *payload = std::move(received);
  return true;
}

bool TpControlChannel::Handshake(const TpControlConfig& config,
                                 std::string* error) {
  if (config.world_size != 2 || config.rank != rank_ || config.rank > 1 ||
      config.max_context == 0 || config.prefill_chunk_tokens == 0 ||
      config.auth_token.empty() ||
      config.auth_token.size() > kMaxAuthTokenBytes || handshaken_ ||
      (config.use_mtp && config.max_draft_tokens == 0)) {
    SetError(error, "TP control configuration is invalid");
    return false;
  }
  auth_token_ = config.auth_token;
  std::vector<std::uint8_t> payload;
  AppendU32(&payload, config.rank);
  AppendU32(&payload, config.world_size);
  AppendU32(&payload, config.max_context);
  AppendU32(&payload, config.prefill_chunk_tokens);
  AppendU32(&payload, config.max_draft_tokens);
  AppendU32(&payload, config.use_mtp ? 1U : 0U);
  AppendU32(&payload, config.allow_cache_reuse ? 1U : 0U);
  AppendU32(&payload, static_cast<std::uint32_t>(config.auth_token.size()));
  payload.insert(payload.end(), config.auth_token.begin(),
                 config.auth_token.end());
  if (!SendFrame(kHello, 0, payload, error)) {
    return false;
  }
  std::uint64_t sequence = 0;
  std::vector<std::uint8_t> peer;
  if (!ReceiveFrame(kHello, &sequence, &peer, error) || sequence != 0) {
    SetError(error, "TP control hello sequence is invalid");
    return false;
  }
  std::size_t offset = 0;
  std::uint32_t rank = 0;
  std::uint32_t world = 0;
  std::uint32_t context = 0;
  std::uint32_t prefill_chunk = 0;
  std::uint32_t draft = 0;
  std::uint32_t mtp = 0;
  std::uint32_t cache_reuse = 0;
  std::uint32_t auth_size = 0;
  std::string peer_token;
  if (!ReadU32(peer, &offset, &rank, error) ||
      !ReadU32(peer, &offset, &world, error) ||
      !ReadU32(peer, &offset, &context, error) ||
      !ReadU32(peer, &offset, &prefill_chunk, error) ||
      !ReadU32(peer, &offset, &draft, error) ||
      !ReadU32(peer, &offset, &mtp, error) ||
      !ReadU32(peer, &offset, &cache_reuse, error) ||
      !ReadU32(peer, &offset, &auth_size, error) ||
      auth_size > kMaxAuthTokenBytes || peer.size() - offset != auth_size) {
    SetError(error, "TP control hello payload is invalid");
    return false;
  }
  peer_token.assign(reinterpret_cast<const char*>(peer.data() + offset),
                    auth_size);
  if (rank != 1U - config.rank || world != config.world_size ||
      context != config.max_context ||
      prefill_chunk != config.prefill_chunk_tokens ||
      draft != config.max_draft_tokens || mtp != (config.use_mtp ? 1U : 0U) ||
      cache_reuse != (config.allow_cache_reuse ? 1U : 0U) ||
      peer_token != auth_token_) {
    SetError(error, "TP control hello configuration mismatch");
    return false;
  }
  world_size_ = config.world_size;
  max_context_ = config.max_context;
  handshaken_ = true;
  ClearReceiveTimeout(fd_);
  return true;
}

bool TpControlChannel::SendCommand(const TpControlCommand& command,
                                   std::string* error) {
  if (!handshaken_ || rank_ != 0 || !ValidateTpControlCommand(command, error)) {
    if (error != nullptr && error->empty()) {
      SetError(error, "TP control command is invalid");
    }
    return false;
  }
  const auto members = EffectiveCommandMembers(command);
  if (max_context_ == 0 ||
      std::ranges::any_of(members, [&](const auto& member) {
        return member.prompt_tokens.size() > max_context_ ||
               (command.kind == TpControlCommandKind::kCohort2Ar &&
                member.max_tokens > max_context_);
      })) {
    SetError(error, "TP control command exceeds the negotiated context");
    return false;
  }

  std::vector<std::uint8_t> payload;
  payload.reserve(96 + members.size() * 64);
  AppendU64(&payload, command.sequence);
  AppendU32(&payload, static_cast<std::uint32_t>(command.kind));
  AppendU64(&payload, command.kind == TpControlCommandKind::kSingle
                          ? command.sequence
                          : command.cohort_id);
  const TpPlanDigest empty_digest{};
  const auto& execution_digest = command.kind == TpControlCommandKind::kSingle
                                     ? empty_digest
                                     : command.execution_plan_digest;
  const auto& cache_digest = command.kind == TpControlCommandKind::kSingle
                                 ? empty_digest
                                 : command.cache_plan_digest;
  AppendBytes(&payload, execution_digest);
  AppendBytes(&payload, cache_digest);
  AppendU32(&payload, static_cast<std::uint32_t>(members.size()));
  for (const auto& member : members) {
    AppendU64(&payload, member.member_id);
    AppendU32(&payload, member.max_tokens);
    AppendU32(&payload, member.cache_prompt ? 1U : 0U);
    AppendU32(&payload, member.cache_prefix_tokens);
    AppendU32(&payload,
              static_cast<std::uint32_t>(member.prompt_tokens.size()));
    AppendU32(&payload, static_cast<std::uint32_t>(member.client_id.size()));
    for (const auto token : member.prompt_tokens) {
      AppendU32(&payload, static_cast<std::uint32_t>(token));
    }
    payload.insert(payload.end(), member.client_id.begin(),
                   member.client_id.end());
  }
  // Kind-specific fields follow the member block. The layout is a function of
  // the kind, which is already on the wire, so both sides derive the same
  // framing without a separate encoder.
  if (command.kind == TpControlCommandKind::kSingle) {
    AppendU32(&payload, command.sampled ? 1U : 0U);
  } else if (command.kind == TpControlCommandKind::kInstruction) {
    const auto& instruction = command.instruction;
    AppendU32(&payload, static_cast<std::uint32_t>(instruction.op));
    AppendU64(&payload, instruction.index);
    AppendU32(&payload, instruction.state);
    AppendU32(&payload, static_cast<std::uint32_t>(instruction.token));
    AppendU32(&payload, instruction.offset);
    AppendU32(&payload, instruction.count);
    AppendU32(&payload, instruction.prompt_size);
    AppendU64(&payload, instruction.digest);
  }
  return SendFrame(kCommand, command.sequence, payload, error);
}

bool TpControlChannel::ReceiveCommand(TpControlCommand* command,
                                      std::string* error) {
  if (!handshaken_ || rank_ != 1 || command == nullptr) {
    SetError(error, "TP control command role is invalid");
    return false;
  }
  *command = {};
  std::uint64_t sequence = 0;
  if (!ReceiveFrame(kCommand, &sequence, &command_prompt_, error)) {
    return false;
  }

  TpControlCommand parsed;
  parsed.sequence = sequence;
  std::size_t offset = 0;
  std::uint64_t embedded = 0;
  std::uint32_t kind = 0;
  std::uint32_t member_count = 0;
  if (!ReadU64(command_prompt_, &offset, &embedded, error) ||
      !ReadU32(command_prompt_, &offset, &kind, error) ||
      !ReadU64(command_prompt_, &offset, &parsed.cohort_id, error) ||
      !ReadBytes(command_prompt_, &offset, parsed.execution_plan_digest,
                 error) ||
      !ReadBytes(command_prompt_, &offset, parsed.cache_plan_digest, error) ||
      !ReadU32(command_prompt_, &offset, &member_count, error) ||
      // An instruction belongs to a request rather than to a member, so it is
      // the one kind that legitimately carries none. The exact per-kind count
      // is enforced once the kind is known, below.
      embedded != sequence ||
      (member_count == 0 && kind != static_cast<std::uint32_t>(
                                        TpControlCommandKind::kInstruction)) ||
      member_count > kMaxCohortMembers) {
    command_prompt_.clear();
    command_prompt_.shrink_to_fit();
    SetError(error, "TP control command envelope is invalid");
    return false;
  }
  if (kind == static_cast<std::uint32_t>(TpControlCommandKind::kSingle)) {
    parsed.kind = TpControlCommandKind::kSingle;
  } else if (kind ==
             static_cast<std::uint32_t>(TpControlCommandKind::kCohort2Ar)) {
    parsed.kind = TpControlCommandKind::kCohort2Ar;
  } else if (kind ==
             static_cast<std::uint32_t>(TpControlCommandKind::kInstruction)) {
    parsed.kind = TpControlCommandKind::kInstruction;
  } else {
    command_prompt_.clear();
    command_prompt_.shrink_to_fit();
    SetError(error, "TP control command kind is invalid");
    return false;
  }
  // An instruction belongs to a request, not to a member, so it carries none.
  // kSingle always has exactly one and kCohort2Ar exactly two.
  const std::size_t expected_members =
      parsed.kind == TpControlCommandKind::kSingle      ? 1
      : parsed.kind == TpControlCommandKind::kCohort2Ar ? kMaxCohortMembers
                                                        : 0;
  if (member_count != expected_members) {
    command_prompt_.clear();
    command_prompt_.shrink_to_fit();
    SetError(error, "TP control command member count is invalid");
    return false;
  }

  std::vector<TpControlMemberRequest> members(member_count);
  for (auto& member : members) {
    std::uint32_t cache_prompt = 0;
    std::uint32_t prompt_count = 0;
    std::uint32_t client_size = 0;
    if (!ReadU64(command_prompt_, &offset, &member.member_id, error) ||
        !ReadU32(command_prompt_, &offset, &member.max_tokens, error) ||
        !ReadU32(command_prompt_, &offset, &cache_prompt, error) ||
        !ReadU32(command_prompt_, &offset, &member.cache_prefix_tokens,
                 error) ||
        !ReadU32(command_prompt_, &offset, &prompt_count, error) ||
        !ReadU32(command_prompt_, &offset, &client_size, error) ||
        cache_prompt > 1 || prompt_count == 0 ||
        prompt_count > kMaxPromptTokens || client_size > kMaxClientIdBytes ||
        client_size > command_prompt_.size() - offset) {
      command_prompt_.clear();
      command_prompt_.shrink_to_fit();
      SetError(error, "TP control command member is invalid");
      return false;
    }
    member.cache_prompt = cache_prompt != 0;
    member.prompt_tokens.resize(prompt_count);
    for (auto& token : member.prompt_tokens) {
      std::uint32_t value = 0;
      if (!ReadU32(command_prompt_, &offset, &value, error)) {
        return false;
      }
      token = static_cast<std::int32_t>(value);
    }
    member.client_id.assign(
        reinterpret_cast<const char*>(command_prompt_.data() + offset),
        client_size);
    offset += client_size;
  }
  if (parsed.kind == TpControlCommandKind::kSingle) {
    std::uint32_t sampled = 0;
    if (!ReadU32(command_prompt_, &offset, &sampled, error) || sampled > 1) {
      command_prompt_.clear();
      command_prompt_.shrink_to_fit();
      SetError(error, "TP C1 command sampling flag is invalid");
      return false;
    }
    parsed.sampled = sampled != 0;
  } else if (parsed.kind == TpControlCommandKind::kInstruction) {
    auto& instruction = parsed.instruction;
    std::uint32_t op = 0;
    std::uint32_t token_bits = 0;
    if (!ReadU32(command_prompt_, &offset, &op, error) ||
        !ReadU64(command_prompt_, &offset, &instruction.index, error) ||
        !ReadU32(command_prompt_, &offset, &instruction.state, error) ||
        !ReadU32(command_prompt_, &offset, &token_bits, error) ||
        !ReadU32(command_prompt_, &offset, &instruction.offset, error) ||
        !ReadU32(command_prompt_, &offset, &instruction.count, error) ||
        !ReadU32(command_prompt_, &offset, &instruction.prompt_size, error) ||
        !ReadU64(command_prompt_, &offset, &instruction.digest, error) ||
        op > static_cast<std::uint32_t>(TpInstructionOp::kEnd)) {
      command_prompt_.clear();
      command_prompt_.shrink_to_fit();
      SetError(error, "TP control instruction is invalid");
      return false;
    }
    instruction.op = static_cast<TpInstructionOp>(op);
    instruction.token = static_cast<std::int32_t>(token_bits);
  }
  if (offset != command_prompt_.size()) {
    command_prompt_.clear();
    command_prompt_.shrink_to_fit();
    SetError(error, "TP control command payload has trailing bytes");
    return false;
  }

  if (parsed.kind == TpControlCommandKind::kSingle) {
    if (parsed.cohort_id != sequence || members.front().member_id != sequence) {
      command_prompt_.clear();
      command_prompt_.shrink_to_fit();
      SetError(error, "TP C1 command scope is not its sequence");
      return false;
    }
    parsed.max_tokens = members.front().max_tokens;
    parsed.cache_prompt = members.front().cache_prompt;
    parsed.cache_prefix_tokens = members.front().cache_prefix_tokens;
    parsed.prompt_tokens = std::move(members.front().prompt_tokens);
    parsed.client_id = std::move(members.front().client_id);
  } else {
    parsed.members = std::move(members);
  }
  if (max_context_ == 0 ||
      std::ranges::any_of(EffectiveCommandMembers(parsed),
                          [&](const auto& member) {
                            return member.prompt_tokens.size() > max_context_ ||
                                   (parsed.kind ==
                                        TpControlCommandKind::kCohort2Ar &&
                                    member.max_tokens > max_context_);
                          }) ||
      !ValidateTpControlCommand(parsed, error)) {
    command_prompt_.clear();
    command_prompt_.shrink_to_fit();
    return false;
  }
  *command = std::move(parsed);
  return true;
}

bool TpControlChannel::SendResponse(const TpControlResponse& response,
                                    std::string* error) {
  if (!handshaken_ || rank_ != 1 ||
      !ValidateTpControlResponse(response, error)) {
    if (error != nullptr && error->empty()) {
      SetError(error, "TP control response is invalid");
    }
    return false;
  }
  std::vector<TpControlMemberResponse> members;
  if (response.kind == TpControlResponseKind::kSingle &&
      response.members.empty()) {
    members.push_back({
        .member_id = response.sequence,
        .tokens = response.tokens,
        .draft_tokens = response.draft_tokens,
        .draft_accepted_tokens = response.draft_accepted_tokens,
        .cached_prompt_tokens = response.cached_prompt_tokens,
        .cache_snapshot_bytes = response.cache_snapshot_bytes,
    });
  } else {
    members = response.members;
  }

  std::vector<std::uint8_t> payload;
  payload.reserve(128 + members.size() * 64 + response.error.size());
  AppendU64(&payload, response.sequence);
  AppendU32(&payload, static_cast<std::uint32_t>(response.kind));
  AppendU64(&payload, response.kind == TpControlResponseKind::kSingle
                          ? response.sequence
                          : response.cohort_id);
  const TpPlanDigest empty_digest{};
  const auto& execution_digest = response.kind == TpControlResponseKind::kSingle
                                     ? empty_digest
                                     : response.execution_plan_digest;
  const auto& cache_digest = response.kind == TpControlResponseKind::kSingle
                                 ? empty_digest
                                 : response.cache_plan_digest;
  AppendBytes(&payload, execution_digest);
  AppendBytes(&payload, cache_digest);
  AppendU32(&payload, static_cast<std::uint32_t>(members.size()));
  for (const auto& member : members) {
    AppendU64(&payload, member.member_id);
    AppendU32(&payload, static_cast<std::uint32_t>(member.tokens.size()));
    AppendU64(&payload, member.draft_tokens);
    AppendU64(&payload, member.draft_accepted_tokens);
    AppendU32(&payload, member.cached_prompt_tokens);
    AppendU64(&payload, member.cache_snapshot_bytes);
    for (const auto token : member.tokens) {
      AppendU32(&payload, static_cast<std::uint32_t>(token));
    }
  }
  AppendU32(&payload, static_cast<std::uint32_t>(response.error.size()));
  payload.insert(payload.end(), response.error.begin(), response.error.end());
  return SendFrame(kResponse, response.sequence, payload, error);
}

bool TpControlChannel::ReceiveResponse(TpControlResponse* response,
                                       std::string* error) {
  if (!handshaken_ || rank_ != 0 || response == nullptr) {
    SetError(error, "TP control response role is invalid");
    return false;
  }
  *response = {};
  std::uint64_t sequence = 0;
  if (!ReceiveFrame(kResponse, &sequence, &response_payload_, error)) {
    return false;
  }

  TpControlResponse parsed;
  parsed.sequence = sequence;
  std::size_t offset = 0;
  std::uint64_t embedded = 0;
  std::uint32_t kind = 0;
  std::uint32_t member_count = 0;
  if (!ReadU64(response_payload_, &offset, &embedded, error) ||
      !ReadU32(response_payload_, &offset, &kind, error) ||
      !ReadU64(response_payload_, &offset, &parsed.cohort_id, error) ||
      !ReadBytes(response_payload_, &offset, parsed.execution_plan_digest,
                 error) ||
      !ReadBytes(response_payload_, &offset, parsed.cache_plan_digest, error) ||
      !ReadU32(response_payload_, &offset, &member_count, error) ||
      embedded != sequence || member_count == 0 ||
      member_count > kMaxCohortMembers) {
    response_payload_.clear();
    response_payload_.shrink_to_fit();
    SetError(error, "TP control response envelope is invalid");
    return false;
  }
  if (kind == static_cast<std::uint32_t>(TpControlResponseKind::kSingle)) {
    parsed.kind = TpControlResponseKind::kSingle;
  } else if (kind ==
             static_cast<std::uint32_t>(TpControlResponseKind::kCohort2Ar)) {
    parsed.kind = TpControlResponseKind::kCohort2Ar;
  } else {
    response_payload_.clear();
    response_payload_.shrink_to_fit();
    SetError(error, "TP control response kind is invalid");
    return false;
  }
  const std::size_t expected_members =
      parsed.kind == TpControlResponseKind::kSingle ? 1 : kMaxCohortMembers;
  if (member_count != expected_members) {
    response_payload_.clear();
    response_payload_.shrink_to_fit();
    SetError(error, "TP control response member count is invalid");
    return false;
  }

  std::vector<TpControlMemberResponse> members(member_count);
  for (auto& member : members) {
    std::uint32_t token_count = 0;
    if (!ReadU64(response_payload_, &offset, &member.member_id, error) ||
        !ReadU32(response_payload_, &offset, &token_count, error) ||
        !ReadU64(response_payload_, &offset, &member.draft_tokens, error) ||
        !ReadU64(response_payload_, &offset, &member.draft_accepted_tokens,
                 error) ||
        !ReadU32(response_payload_, &offset, &member.cached_prompt_tokens,
                 error) ||
        !ReadU64(response_payload_, &offset, &member.cache_snapshot_bytes,
                 error) ||
        token_count > kMaxPromptTokens) {
      response_payload_.clear();
      response_payload_.shrink_to_fit();
      SetError(error, "TP control response member is invalid");
      return false;
    }
    member.tokens.resize(token_count);
    for (auto& token : member.tokens) {
      std::uint32_t value = 0;
      if (!ReadU32(response_payload_, &offset, &value, error)) {
        return false;
      }
      token = static_cast<std::int32_t>(value);
    }
  }
  std::uint32_t error_size = 0;
  if (!ReadU32(response_payload_, &offset, &error_size, error) ||
      error_size > kMaxErrorBytes ||
      response_payload_.size() - offset != error_size) {
    response_payload_.clear();
    response_payload_.shrink_to_fit();
    SetError(error, "TP control response error payload is invalid");
    return false;
  }
  parsed.error.assign(
      reinterpret_cast<const char*>(response_payload_.data() + offset),
      error_size);

  if (parsed.kind == TpControlResponseKind::kSingle) {
    if (parsed.cohort_id != sequence || members.front().member_id != sequence) {
      response_payload_.clear();
      response_payload_.shrink_to_fit();
      SetError(error, "TP C1 response scope is not its sequence");
      return false;
    }
    parsed.tokens = std::move(members.front().tokens);
    parsed.draft_tokens = members.front().draft_tokens;
    parsed.draft_accepted_tokens = members.front().draft_accepted_tokens;
    parsed.cached_prompt_tokens = members.front().cached_prompt_tokens;
    parsed.cache_snapshot_bytes = members.front().cache_snapshot_bytes;
  } else {
    parsed.members = std::move(members);
  }
  if (!ValidateTpControlResponse(parsed, error)) {
    response_payload_.clear();
    response_payload_.shrink_to_fit();
    return false;
  }
  *response = std::move(parsed);
  return true;
}

struct TpResponseBroker::Impl {
  struct Pending {
    TpControlResponse response;
    std::optional<TpResponseExpectation> expectation;
    bool ready{false};
    bool delivered{false};
  };

  [[nodiscard]] static bool MatchesExpectation(
      const TpControlResponse& response,
      const TpResponseExpectation& expectation) noexcept {
    if (response.kind != TpControlResponseKind::kCohort2Ar ||
        response.cohort_id != expectation.cohort_id ||
        response.execution_plan_digest != expectation.execution_plan_digest ||
        response.cache_plan_digest != expectation.cache_plan_digest ||
        response.members.size() != expectation.member_ids.size()) {
      return false;
    }
    for (std::size_t index = 0; index < response.members.size(); ++index) {
      if (response.members[index].member_id != expectation.member_ids[index]) {
        return false;
      }
    }
    return true;
  }

  Impl(std::shared_ptr<TpControlChannel> channel, std::size_t capacity)
      : control(std::move(channel)), capacity(capacity) {
    if (!control || control->rank() != 0 || capacity == 0) {
      throw std::invalid_argument(
          "TP response broker requires a rank-zero channel and capacity");
    }
    reader = std::thread([this] { ReadLoop(); });
  }

  ~Impl() { Stop("TP response broker destroyed"); }

  void Stop(std::string reason) {
    const std::lock_guard<std::mutex> stop_lock(stop_mutex);
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!stopping) {
        stopping = true;
        poisoned = true;
        failure = std::move(reason);
      }
      condition.notify_all();
    }
    if (control) {
      control->Interrupt();
    }
    if (reader.joinable()) {
      reader.join();
    }
  }

  void ReadLoop() {
    for (;;) {
      TpControlResponse response;
      std::string error;
      if (!control->ReceiveResponse(&response, &error)) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!stopping) {
          stopping = true;
          poisoned = true;
          failure = "TP worker response receive failed: " + error;
        }
        condition.notify_all();
        return;
      }
      {
        std::lock_guard<std::mutex> lock(mutex);
        if (stopping) {
          return;
        }
        const auto found = pending.find(response.sequence);
        if (found == pending.end() || found->second->ready) {
          stopping = true;
          poisoned = true;
          failure = "TP response sequence is unknown or duplicate: " +
                    std::to_string(response.sequence);
          condition.notify_all();
          control->Interrupt();
          return;
        }
        if (found->second->expectation &&
            !MatchesExpectation(response, *found->second->expectation)) {
          stopping = true;
          poisoned = true;
          failure = "TP response cohort contract mismatch: " +
                    std::to_string(response.sequence);
          condition.notify_all();
          control->Interrupt();
          return;
        }
        found->second->response = std::move(response);
        found->second->ready = true;
        condition.notify_all();
      }
    }
  }

  std::shared_ptr<TpControlChannel> control;
  const std::size_t capacity;
  std::mutex mutex;
  std::condition_variable condition;
  std::mutex stop_mutex;
  std::unordered_map<std::uint64_t, std::shared_ptr<Pending>> pending;
  bool stopping{false};
  bool poisoned{false};
  std::string failure;
  std::thread reader;
};

TpResponseBroker::TpResponseBroker(std::shared_ptr<TpControlChannel> control,
                                   std::size_t max_pending_responses)
    : impl_(std::make_unique<Impl>(std::move(control), max_pending_responses)) {
}

TpResponseBroker::~TpResponseBroker() {
  impl_.reset();
}

bool TpResponseBroker::RegisterPendingResponse(std::uint64_t sequence,
                                               std::string* error) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->stopping || impl_->poisoned) {
    SetError(error, impl_->failure.empty() ? "TP response broker is stopped"
                                           : impl_->failure);
    return false;
  }
  if (impl_->pending.size() >= impl_->capacity) {
    SetError(error, "TP response broker capacity is exhausted");
    return false;
  }
  if (!impl_->pending.emplace(sequence, std::make_shared<Impl::Pending>())
           .second) {
    SetError(error, "TP response sequence is already registered");
    return false;
  }
  return true;
}

bool TpResponseBroker::RegisterPendingResponse(
    std::uint64_t sequence, const TpResponseExpectation& expectation,
    std::string* error) {
  if (expectation.cohort_id == 0 ||
      IsZeroDigest(expectation.execution_plan_digest) ||
      IsZeroDigest(expectation.cache_plan_digest) ||
      expectation.member_ids.size() != kMaxCohortMembers) {
    SetError(error, "TP C2 response expectation is invalid");
    return false;
  }
  for (std::size_t index = 0; index < expectation.member_ids.size(); ++index) {
    if (expectation.member_ids[index] == 0) {
      SetError(error, "TP C2 response expectation has a zero member ID");
      return false;
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (expectation.member_ids[previous] == expectation.member_ids[index]) {
        SetError(error, "TP C2 response expectation member IDs must be unique");
        return false;
      }
    }
  }

  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->stopping || impl_->poisoned) {
    SetError(error, impl_->failure.empty() ? "TP response broker is stopped"
                                           : impl_->failure);
    return false;
  }
  if (impl_->pending.size() >= impl_->capacity) {
    SetError(error, "TP response broker capacity is exhausted");
    return false;
  }
  auto pending = std::make_shared<Impl::Pending>();
  pending->expectation = expectation;
  if (!impl_->pending.emplace(sequence, std::move(pending)).second) {
    SetError(error, "TP response sequence is already registered");
    return false;
  }
  return true;
}

bool TpResponseBroker::CancelUnsentResponse(std::uint64_t sequence,
                                            std::string* error) {
  bool interrupt = false;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto found = impl_->pending.find(sequence);
    if (found == impl_->pending.end()) {
      SetError(error, "TP response sequence is not registered");
      return false;
    }
    if (found->second->ready) {
      impl_->stopping = true;
      impl_->poisoned = true;
      impl_->failure = "TP response arrived before command send completed: " +
                       std::to_string(sequence);
      impl_->condition.notify_all();
      SetError(error, impl_->failure);
      interrupt = true;
    } else {
      impl_->pending.erase(found);
    }
  }
  if (interrupt) {
    impl_->control->Interrupt();
  }
  return !interrupt;
}

bool TpResponseBroker::WaitForResponse(std::uint64_t sequence,
                                       TpControlResponse* response,
                                       std::string* error) {
  if (response == nullptr) {
    SetError(error, "TP response output is null");
    return false;
  }
  std::unique_lock<std::mutex> lock(impl_->mutex);
  const auto found = impl_->pending.find(sequence);
  if (found == impl_->pending.end()) {
    SetError(error, "TP response sequence is not registered");
    return false;
  }
  const auto pending = found->second;
  impl_->condition.wait(lock, [&] {
    return pending->ready || impl_->poisoned || impl_->stopping;
  });
  if (pending->delivered) {
    SetError(error, "TP response sequence was already consumed");
    return false;
  }
  if (impl_->poisoned || impl_->stopping) {
    SetError(error, impl_->failure.empty() ? "TP response broker is stopped"
                                           : impl_->failure);
    return false;
  }
  if (pending->ready) {
    pending->delivered = true;
    *response = std::move(pending->response);
    impl_->pending.erase(found);
    return true;
  }
  SetError(error, impl_->failure.empty() ? "TP response broker is stopped"
                                         : impl_->failure);
  return false;
}

void TpResponseBroker::FailAll(std::string reason) {
  if (impl_) {
    impl_->Stop(std::move(reason));
  }
}

std::uint16_t TpControlChannel::port() const noexcept {
  return port_;
}
std::uint32_t TpControlChannel::rank() const noexcept {
  return rank_;
}
std::uint32_t TpControlChannel::world_size() const noexcept {
  return world_size_;
}

}  // namespace gufo::server

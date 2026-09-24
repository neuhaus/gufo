#include "src/cli/serve/tp_control.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <limits>
#include <span>
#include <string_view>
#include <thread>
#include <utility>

namespace gufo::server {
namespace {

constexpr std::uint32_t kMagic = 0x54504331U;  // "TPC1"
constexpr std::uint16_t kVersion = 1;
constexpr std::uint16_t kHello = 1;
constexpr std::uint16_t kCommand = 2;
constexpr std::uint16_t kResponse = 3;
constexpr std::size_t kMaxPayloadBytes = 16U << 20;
constexpr std::size_t kMaxPromptTokens = 1U << 20;
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

void SetOperationTimeouts(int fd) {
  timeval timeout{};
  timeout.tv_sec = 24 * 60 * 60;
  (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
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
  if (*offset > data.size() ||
      data.size() - *offset < sizeof(std::uint32_t)) {
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
  if (*offset > data.size() ||
      data.size() - *offset < sizeof(std::uint64_t)) {
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

}  // namespace

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
    std::uint16_t port, std::string* error) {
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
  SetOperationTimeouts(peer);
  return std::shared_ptr<TpControlChannel>(
      new TpControlChannel(peer, 0, port));
}

std::shared_ptr<TpControlChannel> TpControlChannel::Connect(
    const std::string& host, std::uint16_t port, std::string* error) {
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
    const int resolve = ::getaddrinfo(host.c_str(), service.c_str(), &hints,
                                      &addresses);
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
      SetOperationTimeouts(fd);
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
  if (fd_ >= 0) {
    ::close(fd_);
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
  const std::lock_guard<std::mutex> lock(io_mutex_);
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
  const std::lock_guard<std::mutex> lock(io_mutex_);
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
  std::uint32_t auth_size = 0;
  std::string peer_token;
  if (!ReadU32(peer, &offset, &rank, error) ||
      !ReadU32(peer, &offset, &world, error) ||
      !ReadU32(peer, &offset, &context, error) ||
      !ReadU32(peer, &offset, &prefill_chunk, error) ||
      !ReadU32(peer, &offset, &draft, error) ||
      !ReadU32(peer, &offset, &mtp, error) ||
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
      draft != config.max_draft_tokens ||
      mtp != (config.use_mtp ? 1U : 0U) || peer_token != auth_token_) {
    SetError(error, "TP control hello configuration mismatch");
    return false;
  }
  world_size_ = config.world_size;
  max_context_ = config.max_context;
  handshaken_ = true;
  return true;
}

bool TpControlChannel::SendCommand(const TpControlCommand& command,
                                   std::string* error) {
  if (!handshaken_ || rank_ != 0 || command.max_tokens == 0 ||
      command.prompt_tokens.empty() ||
      command.prompt_tokens.size() > kMaxPromptTokens ||
      command.client_id.size() > kMaxClientIdBytes) {
    SetError(error, "TP control command is invalid");
    return false;
  }
  std::vector<std::uint8_t> payload;
  payload.reserve(20 + command.prompt_tokens.size() * sizeof(std::int32_t) +
                  command.client_id.size());
  AppendU64(&payload, command.sequence);
  AppendU32(&payload, command.max_tokens);
  AppendU32(&payload, static_cast<std::uint32_t>(command.prompt_tokens.size()));
  AppendU32(&payload, static_cast<std::uint32_t>(command.client_id.size()));
  for (const auto token : command.prompt_tokens) {
    AppendU32(&payload, static_cast<std::uint32_t>(token));
  }
  payload.insert(payload.end(), command.client_id.begin(),
                 command.client_id.end());
  return SendFrame(kCommand, command.sequence, payload, error);
}

bool TpControlChannel::ReceiveCommand(TpControlCommand* command,
                                      std::string* error) {
  if (!handshaken_ || rank_ != 1 || command == nullptr) {
    SetError(error, "TP control command role is invalid");
    return false;
  }
  std::uint64_t sequence = 0;
  if (!ReceiveFrame(kCommand, &sequence, &command_prompt_, error)) {
    return false;
  }
  std::size_t offset = 0;
  std::uint64_t embedded = 0;
  std::uint32_t max_tokens = 0;
  std::uint32_t prompt_count = 0;
  std::uint32_t client_size = 0;
  if (!ReadU64(command_prompt_, &offset, &embedded, error) ||
      !ReadU32(command_prompt_, &offset, &max_tokens, error) ||
      !ReadU32(command_prompt_, &offset, &prompt_count, error) ||
      !ReadU32(command_prompt_, &offset, &client_size, error) ||
      embedded != sequence || max_tokens == 0 ||
      prompt_count == 0 || prompt_count > kMaxPromptTokens ||
      max_context_ == 0 || prompt_count > max_context_ ||
      client_size > kMaxClientIdBytes ||
      command_prompt_.size() - offset !=
          static_cast<std::size_t>(prompt_count) * sizeof(std::int32_t) +
              client_size) {
    command_prompt_.clear();
    command_prompt_.shrink_to_fit();
    SetError(error, "TP control command payload is invalid");
    return false;
  }
  command->sequence = sequence;
  command->max_tokens = max_tokens;
  command->prompt_tokens.resize(prompt_count);
  for (auto& token : command->prompt_tokens) {
    std::uint32_t value = 0;
    if (!ReadU32(command_prompt_, &offset, &value, error)) {
      return false;
    }
    token = static_cast<std::int32_t>(value);
  }
  command->client_id.assign(
      reinterpret_cast<const char*>(command_prompt_.data() + offset),
      client_size);
  return true;
}

bool TpControlChannel::SendResponse(const TpControlResponse& response,
                                    std::string* error) {
  if (!handshaken_ || rank_ != 1 || response.tokens.size() > kMaxPromptTokens ||
      response.error.size() > kMaxErrorBytes) {
    SetError(error, "TP control response is invalid");
    return false;
  }
  std::vector<std::uint8_t> payload;
  payload.reserve(28 + response.tokens.size() * sizeof(std::int32_t) +
                  response.error.size());
  AppendU64(&payload, response.sequence);
  AppendU32(&payload, static_cast<std::uint32_t>(response.tokens.size()));
  AppendU64(&payload, response.draft_tokens);
  AppendU64(&payload, response.draft_accepted_tokens);
  AppendU32(&payload, static_cast<std::uint32_t>(response.error.size()));
  for (const auto token : response.tokens) {
    AppendU32(&payload, static_cast<std::uint32_t>(token));
  }
  payload.insert(payload.end(), response.error.begin(), response.error.end());
  return SendFrame(kResponse, response.sequence, payload, error);
}

bool TpControlChannel::ReceiveResponse(TpControlResponse* response,
                                       std::string* error) {
  if (!handshaken_ || rank_ != 0 || response == nullptr) {
    SetError(error, "TP control response role is invalid");
    return false;
  }
  std::uint64_t sequence = 0;
  if (!ReceiveFrame(kResponse, &sequence, &response_payload_, error)) {
    return false;
  }
  std::size_t offset = 0;
  std::uint64_t embedded = 0;
  std::uint32_t token_count = 0;
  std::uint32_t error_size = 0;
  if (!ReadU64(response_payload_, &offset, &embedded, error) ||
      !ReadU32(response_payload_, &offset, &token_count, error) ||
      !ReadU64(response_payload_, &offset, &response->draft_tokens, error) ||
      !ReadU64(response_payload_, &offset, &response->draft_accepted_tokens,
               error) ||
      !ReadU32(response_payload_, &offset, &error_size, error) ||
      embedded != sequence || token_count > kMaxPromptTokens ||
      error_size > kMaxErrorBytes ||
      response_payload_.size() - offset !=
          static_cast<std::size_t>(token_count) * sizeof(std::int32_t) +
              error_size) {
    response_payload_.clear();
    response_payload_.shrink_to_fit();
    SetError(error, "TP control response payload is invalid");
    return false;
  }
  response->sequence = sequence;
  response->tokens.resize(token_count);
  for (auto& token : response->tokens) {
    std::uint32_t value = 0;
    if (!ReadU32(response_payload_, &offset, &value, error)) {
      return false;
    }
    token = static_cast<std::int32_t>(value);
  }
  response->error.assign(
      reinterpret_cast<const char*>(response_payload_.data() + offset),
      error_size);
  return true;
}

std::uint16_t TpControlChannel::port() const noexcept { return port_; }
std::uint32_t TpControlChannel::rank() const noexcept { return rank_; }
std::uint32_t TpControlChannel::world_size() const noexcept {
  return world_size_;
}

}  // namespace gufo::server

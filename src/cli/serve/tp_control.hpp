#ifndef GUFO_SERVER_TP_CONTROL_HPP_
#define GUFO_SERVER_TP_CONTROL_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"

namespace gufo::server {

struct TpControlConfig {
  std::uint64_t snapshot_budget_bytes{0};
  std::uint32_t rank{0};
  std::uint32_t world_size{1};
  std::uint32_t max_context{4096};
  std::uint32_t max_draft_tokens{7};
  bool use_mtp{false};
  bool allow_cache_reuse{false};
  std::string auth_token;
  std::uint32_t prefill_chunk_tokens{512};
};

enum class TpControlCommandKind : std::uint8_t {
  kSingle = 1,
  /// Control-plane contract only; no scheduler execution path consumes C2 yet.
  kCohort2Ar = 2,
  /// One model call for rank 1 to execute: rank 0's runner sends it just
  /// before running the same call itself, so both ranks make identical model
  /// calls in identical order. See `TpInstruction`.
  kInstruction = 3,
};

/// The model call an instruction asks rank 1 to make.
enum class TpInstructionOp : std::uint8_t {
  kNone = 0,
  /// Reset the state's model session. The continuation cache decides this, so
  /// it may arrive between requests (sequence 0) as well as within one.
  kInvalidate = 1,
  /// `Prefill(state, prompt[0, prompt_size), offset, count)`, where the prompt
  /// is the request's `kSingle` prompt.
  kPrefill = 2,
  /// `Advance(state, token)`.
  kAdvance = 3,
  /// `DecodeStep(state, count)`: one MTP draft-verify cycle. It carries rank
  /// 0's sampler draw state (`rng`, `pending`) from just before the call, so
  /// rank 1's draws, and therefore its decisions, are rank 0's.
  kDecode = 4,
  /// The request is over. `count` is the number of instructions rank 0 sent
  /// for it and `digest` its execution digest; rank 1 compares both.
  kEnd = 5,
  kSnapshot = 6,
  kDrop = 7,
  kRestore = 8,
  kReuse = 9,
  kCancelPrepare = 10,
};

struct TpInstruction {
  TpInstructionOp op{TpInstructionOp::kNone};
  /// Channel-wide and monotonic, so rank 1 detects a lost or repeated frame.
  std::uint64_t index{0};
  std::uint32_t state{0};
  std::int32_t token{0};
  std::uint32_t offset{0};
  std::uint32_t count{0};
  std::uint32_t prompt_size{0};
  std::uint64_t digest{0};
  /// kDecode only: the sampler's RNG state and pending deferred draw (-1 for
  /// none) before the call.
  std::uint64_t rng{0};
  std::int32_t pending{-1};
  std::uint64_t snapshot_id{0};

  bool operator==(const TpInstruction&) const = default;
};

enum class TpControlResponseKind : std::uint8_t {
  kSingle = 1,
  /// Completion of a cache operation, before either rank can forward again.
  kInstruction = 3,
  /// Response envelope for the dormant C2 control contract.
  kCohort2Ar = 2,
};

using TpPlanDigest = std::array<std::uint8_t, 32>;

struct TpControlMemberRequest {
  std::uint64_t member_id{0};
  std::uint32_t max_tokens{0};
  bool cache_prompt{false};
  std::uint32_t cache_prefix_tokens{0};
  std::vector<std::int32_t> prompt_tokens;
  std::string client_id;
};

struct TpControlMemberResponse {
  std::uint64_t member_id{0};
  std::vector<std::int32_t> tokens;
  std::uint64_t draft_tokens{0};
  std::uint64_t draft_accepted_tokens{0};
  std::uint32_t cached_prompt_tokens{0};
  std::uint64_t cache_snapshot_bytes{0};
};

struct TpControlCommand {
  /// Response correlation key. For C1 this remains the collective scope.
  std::uint64_t sequence{0};
  // Fixed one-member C1 fields. kCohort2Ar leaves these empty and uses members.
  std::uint32_t max_tokens{0};
  bool cache_prompt{false};
  std::uint32_t cache_prefix_tokens{0};
  std::vector<std::int32_t> prompt_tokens;
  std::string client_id;
  // Versioned cohort envelope. C1 defaults to one member and sequence scope.
  TpControlCommandKind kind{TpControlCommandKind::kSingle};
  std::uint64_t cohort_id{0};
  TpPlanDigest execution_plan_digest{};
  TpPlanDigest cache_plan_digest{};
  std::vector<TpControlMemberRequest> members;
  /// kSingle only: the request's sampling configuration. Rank 1 builds the
  /// same sampler for multi-token decoding, and checks rank 0's tokens against
  /// its own choice only when this is greedy.
  sampling::SamplingConfig sampling{};
  /// kInstruction only. An instruction belongs to the `kSingle` request named
  /// by `sequence` (or to none, for a reset between requests) and carries no
  /// request fields of its own.
  TpInstruction instruction{};
};

struct TpControlResponse {
  std::uint64_t instruction_index{0};
  /// Response broker correlation key. C1 worker fields remain
  /// source-compatible.
  std::uint64_t sequence{0};
  std::vector<std::int32_t> tokens;
  std::uint64_t draft_tokens{0};
  std::uint64_t draft_accepted_tokens{0};
  std::uint32_t cached_prompt_tokens{0};
  std::uint64_t cache_snapshot_bytes{0};
  std::string error;
  TpControlResponseKind kind{TpControlResponseKind::kSingle};
  std::uint64_t cohort_id{0};
  TpPlanDigest execution_plan_digest{};
  TpPlanDigest cache_plan_digest{};
  std::vector<TpControlMemberResponse> members;
};

struct TpResponseExpectation {
  std::uint64_t cohort_id{0};
  TpPlanDigest execution_plan_digest{};
  TpPlanDigest cache_plan_digest{};
  std::vector<std::uint64_t> member_ids;
};

/// Computes the canonical C1/C2 execution-plan identity. C2 includes member
/// order, token budgets and prompt tokens; rank identity is excluded.
/// A `kInstruction` carries no plan: the request it belongs to already agreed
/// one, and validation requires its plan digests to be zero, so a plan digest
/// computed over an instruction is meaningless and never compared.
///
/// This is a command-integrity digest, not a plan-agreement check: both ranks
/// derive it from the same command bytes, so a match proves the command was not
/// mutated but says nothing about whether the two ranks will execute it the
/// same way. In particular it cannot distinguish a serial cohort from a batched
/// one, because the execution width is not carried on the wire. The shape
/// fields are not individually named because their meaning was never
/// established; do not infer semantics from their positions.
[[nodiscard]] TpPlanDigest ComputeTpExecutionPlanDigest(
    const TpControlCommand& command);

/// Computes the canonical requested cache plan. The bounded C2 contract only
/// accepts the all-disabled plan, but C1 retains its existing cache fields.
[[nodiscard]] TpPlanDigest ComputeTpCachePlanDigest(
    const TpControlCommand& command);

[[nodiscard]] bool ValidateTpControlCommand(const TpControlCommand& command,
                                            std::string* error);
[[nodiscard]] bool ValidateTpControlResponse(const TpControlResponse& response,
                                             std::string* error);

/// Default bound on a blocking control-channel send, and on a receive before
/// the handshake completes. It is generous because a rank waits in the
/// handshake while its peer is still loading the model.
inline constexpr std::chrono::milliseconds kTpControlIoTimeout =
    std::chrono::hours(24);

/// Ordered, versioned TCP control channel for the first TP=2 worker slice.
/// Tensor payload still uses the RDMA communicator; this channel carries only
/// prepared prompt/cohort commands, responses, and lifecycle handshakes.
///
/// `io_timeout` bounds every blocking send, and every receive until the
/// handshake succeeds. After the handshake receives have no timeout: a worker
/// waits for its next command, and rank 0's reader for its next response, for
/// as long as the server stays idle. TCP keepalive still reports a peer whose
/// host died.
class TpControlChannel final {
public:
  [[nodiscard]] static std::shared_ptr<TpControlChannel> Listen(
      std::uint16_t port, std::string* error,
      std::chrono::milliseconds io_timeout = kTpControlIoTimeout);
  [[nodiscard]] static std::shared_ptr<TpControlChannel> Connect(
      const std::string& host, std::uint16_t port, std::string* error,
      std::chrono::milliseconds io_timeout = kTpControlIoTimeout);

  ~TpControlChannel();

  TpControlChannel(const TpControlChannel&) = delete;
  TpControlChannel& operator=(const TpControlChannel&) = delete;

  [[nodiscard]] bool Handshake(const TpControlConfig& config,
                               std::string* error);
  [[nodiscard]] bool SendCommand(const TpControlCommand& command,
                                 std::string* error);
  [[nodiscard]] bool ReceiveCommand(TpControlCommand* command,
                                    std::string* error);
  [[nodiscard]] bool SendResponse(const TpControlResponse& response,
                                  std::string* error);
  [[nodiscard]] bool ReceiveResponse(TpControlResponse* response,
                                     std::string* error);
  /// Wake a blocked directional reader/writer during broker shutdown.
  void Interrupt() noexcept;

  [[nodiscard]] std::uint64_t snapshot_budget_bytes() const noexcept {
    return snapshot_budget_bytes_;
  }
  [[nodiscard]] std::uint16_t port() const noexcept;
  [[nodiscard]] std::uint32_t rank() const noexcept;
  [[nodiscard]] std::uint32_t world_size() const noexcept;

private:
  TpControlChannel(int fd, std::uint32_t rank, std::uint16_t port);

  [[nodiscard]] bool SendFrame(std::uint16_t type, std::uint64_t sequence,
                               const std::vector<std::uint8_t>& payload,
                               std::string* error);
  [[nodiscard]] bool ReceiveFrame(std::uint16_t type, std::uint64_t* sequence,
                                  std::vector<std::uint8_t>* payload,
                                  std::string* error);
  [[nodiscard]] bool SendAll(const void* data, std::size_t bytes,
                             std::string* error);
  [[nodiscard]] bool RecvAll(void* data, std::size_t bytes, std::string* error);

  int fd_{-1};
  std::uint16_t port_{0};
  std::uint32_t rank_{0};
  std::uint32_t world_size_{1};
  std::uint32_t max_context_{0};
  bool handshaken_{false};
  std::uint64_t snapshot_budget_bytes_{0};
  std::string auth_token_;
  std::mutex send_mutex_;
  std::mutex receive_mutex_;
  std::atomic<bool> interrupted_{false};
  std::vector<std::uint8_t> command_prompt_;
  std::vector<std::uint8_t> response_payload_;
};

/// Rank-zero response owner. The dedicated reader routes final responses by
/// control sequence; production construction uses capacity one. This is a
/// safety foundation for ordered C2 cohorts, not an enablement of C2 itself.
class TpResponseBroker final {
public:
  TpResponseBroker(std::shared_ptr<TpControlChannel> control,
                   std::size_t max_pending_responses);
  ~TpResponseBroker();

  TpResponseBroker(const TpResponseBroker&) = delete;
  TpResponseBroker& operator=(const TpResponseBroker&) = delete;
  TpResponseBroker(TpResponseBroker&&) = delete;
  TpResponseBroker& operator=(TpResponseBroker&&) = delete;

  [[nodiscard]] bool RegisterInstruction(std::uint64_t sequence,
                                         std::uint64_t index,
                                         std::string* error);
  [[nodiscard]] bool WaitForInstruction(std::string* error);

  [[nodiscard]] bool RegisterPendingResponse(std::uint64_t sequence,
                                             std::string* error);
  /// Registers one C2 cohort response and validates its envelope before it is
  /// delivered. The existing overload retains one-member C1 compatibility.
  [[nodiscard]] bool RegisterPendingResponse(
      std::uint64_t sequence, const TpResponseExpectation& expectation,
      std::string* error);
  [[nodiscard]] bool CancelUnsentResponse(std::uint64_t sequence,
                                          std::string* error);
  [[nodiscard]] bool WaitForResponse(std::uint64_t sequence,
                                     TpControlResponse* response,
                                     std::string* error);
  void FailAll(std::string reason);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_TP_CONTROL_HPP_

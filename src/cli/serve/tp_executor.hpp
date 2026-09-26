#ifndef GUFO_SERVER_TP_EXECUTOR_HPP_
#define GUFO_SERVER_TP_EXECUTOR_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/cli/serve/text_model_runner.hpp"
#include "src/cli/serve/tp_control.hpp"

namespace gufo::server {

/// Running digest of the model calls one TP2 request made and their results.
///
/// Rank 0 adds each call as it runs it and rank 1 as it executes the same
/// instruction, through the same functions below, so a difference means the
/// ranks ran different calls or got different results. FNV-1a detects
/// accidents; it is not meant to resist an adversary.
class TpExecutionDigest {
public:
  void Add(std::uint64_t value) noexcept;
  void Add(std::span<const TextRunnerToken> tokens) noexcept;
  [[nodiscard]] std::uint64_t value() const noexcept { return value_; }

private:
  std::uint64_t value_{0xcbf29ce484222325ULL};
};

/// The instruction's operation and arguments, before the call runs. The index
/// and the `kEnd` digest are transport fields and are not part of the digest.
void DigestTpCall(TpExecutionDigest& digest, const TpInstruction& instruction,
                  std::span<const TextRunnerToken> prefill_prompt = {});
/// What a call produced, including the state's checkpoint position after it.
void DigestTpPrefill(TpExecutionDigest& digest, const TextPrefillStep& step,
                     std::size_t checkpoint);
void DigestTpAdvance(TpExecutionDigest& digest, std::size_t checkpoint);
void DigestTpDecode(TpExecutionDigest& digest, const TextDecodeStep& step,
                    std::size_t checkpoint);
/// A call that threw. Both ranks record a deterministic failure identically.
void DigestTpFailure(TpExecutionDigest& digest);

/// Where rank 0's instructions go: the control channel in production, a
/// capture in tests.
bool TpCacheAcknowledged(TpInstructionOp op) noexcept;

class TpInstructionSink {
public:
  virtual ~TpInstructionSink() = default;
  virtual void Synchronize(std::uint64_t sequence, TpInstruction instruction,
                           const std::function<void()>& local) = 0;
  /// Sends one instruction for `sequence` (zero between requests) and assigns
  /// its channel-wide index.
  [[nodiscard]] virtual bool Send(std::uint64_t sequence,
                                  TpInstruction instruction,
                                  std::string* error) = 0;
};

class TpControlInstructionSink final : public TpInstructionSink {
public:
  explicit TpControlInstructionSink(std::shared_ptr<TpControlChannel> channel,
                                    std::shared_ptr<TpResponseBroker> broker)
      : channel_(std::move(channel)), broker_(std::move(broker)) {}
  void Synchronize(std::uint64_t sequence, TpInstruction instruction,
                   const std::function<void()>& local) override;

  [[nodiscard]] bool Send(std::uint64_t sequence, TpInstruction instruction,
                          std::string* error) override;

private:
  std::shared_ptr<TpControlChannel> channel_;
  std::shared_ptr<TpResponseBroker> broker_;
  std::mutex mutex_;
  std::uint64_t next_index_{0};
};

/// Rank 0's runner in a TP2 pair.
///
/// Runs every call on the wrapped runner. Just before each call that changes
/// model state it sends the call to rank 1 as an instruction, so rank 1 makes
/// the same call on the same state while rank 0 makes it; both then meet in the
/// call's exchanges. Selection (`SelectNext`, `PreviewFirstToken`) reads only
/// logits and stays local. The scheduler above is unchanged, so its stop,
/// cancellation and length decisions reach rank 1 simply as the calls rank 0
/// no longer makes.
///
/// Cache operations are acknowledged before the next forward. Snapshot bytes
/// stay local; only their monotonic IDs cross the control channel. It never
/// forwards a request's cancellation check to the model session: a session
/// aborting between layers would strand rank 1 inside an exchange, so
/// cancellation acts only between calls, where the scheduler already checks.
class TpMirroredRunner final
    : public TextModelRunner,
      public std::enable_shared_from_this<TpMirroredRunner> {
public:
  TpMirroredRunner(
      std::shared_ptr<TextModelRunner> inner,
      std::shared_ptr<TpInstructionSink> sink,
      std::size_t snapshot_budget = std::numeric_limits<std::size_t>::max());

  /// Attributes the following model calls to `sequence`, the request rank 1
  /// was just told about. Requests do not overlap.
  void BeginRequest(std::uint64_t sequence);
  /// Sends `kEnd` with the request's instruction count and digest and stops
  /// attributing calls to it. Returns false when the send failed.
  [[nodiscard]] bool EndRequest(std::string* error);

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override;
  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override;
  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override;
  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override;
  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override;
  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override;
  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override;
  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override;
  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override;
  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override;
  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override;
  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override;
  void PrepareBatchExecution(TextRunnerState& state) const override;
  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override;
  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override;
  void Advance(TextRunnerState& state, TextRunnerToken token) const override;
  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override;
  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override;
  void AdvanceBatch(std::span<const TextRunnerAdvance> advances) const override;
  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override;
  void PrepareCancellation(TextRunnerState& state) const override;
  std::size_t SnapshotPayloadBytes(const TextRunnerState& state) const override;
  std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override;
  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override;

private:
  class State;
  class SnapshotHandle;
  void Drop(std::uint64_t id) const noexcept;
  void CacheCall(const TpInstruction& instruction,
                 const std::function<void()>& local) const;

  [[nodiscard]] static State& Mirrored(TextRunnerState& state);
  [[nodiscard]] static const State& Mirrored(const TextRunnerState& state);
  /// Records and sends an instruction for the current request. Throws when no
  /// request is active or the channel has failed; the call must not run then,
  /// because rank 1 would never join its exchanges.
  void Send(const TpInstruction& instruction,
            std::span<const TextRunnerToken> prefill_prompt = {}) const;
  /// Sends a reset, which may happen between requests too. Cannot throw: the
  /// continuation cache resets states from `noexcept` paths, so a failed send
  /// is kept and fails the next call instead.
  void SendInvalidate(std::uint32_t state) const noexcept;
  void Record(const std::function<void(TpExecutionDigest&)>& add) const;

  std::shared_ptr<TextModelRunner> inner_;
  std::shared_ptr<TpInstructionSink> sink_;
  const std::size_t snapshot_budget_;
  mutable std::uint64_t next_snapshot_id_{1};
  mutable std::recursive_mutex call_mutex_;
  bool multi_token_decode_{false};
  mutable std::mutex mutex_;
  mutable std::uint32_t next_state_id_{0};
  mutable std::uint64_t sequence_{0};
  mutable std::uint64_t count_{0};
  mutable TpExecutionDigest digest_;
  mutable std::string failure_;
};

/// Rank 1's side of a TP2 pair: executes rank 0's instructions on its own
/// states, which it creates up front in the same number and order as rank 0's
/// pool, so a state id names corresponding states on both ranks.
class TpExecutor {
public:
  TpExecutor(
      std::shared_ptr<TextModelRunner> runner, std::size_t state_count,
      std::size_t snapshot_budget = std::numeric_limits<std::size_t>::max());

  using Acknowledge =
      std::function<bool(const TpControlResponse&, std::string*)>;
  [[nodiscard]] std::size_t snapshot_count() const { return snapshots_.size(); }

  using Receive = std::function<bool(TpControlCommand*, std::string*)>;

  /// Executes an instruction that arrived outside a request (only a reset).
  /// False is a protocol violation: the pair can no longer be trusted.
  [[nodiscard]] bool ExecuteIdle(const TpControlCommand& command,
                                 std::string* error);

  /// Executes one request, `begin` being its `kSingle` command, pulling
  /// instructions from `receive` until `kEnd`. Returns false when the channel
  /// failed or the stream broke the protocol; otherwise `*outcome` is empty
  /// when both ranks agree and describes the first difference or failure.
  [[nodiscard]] bool RunRequest(const TpControlCommand& begin,
                                const Receive& receive,
                                const Acknowledge& acknowledge,
                                std::string* outcome, std::string* error);

private:
  struct OwnChoice {
    bool stop{false};
    TextRunnerToken token{0};
  };

  [[nodiscard]] bool TakeIndex(const TpControlCommand& command,
                               std::string* error);
  [[nodiscard]] TextRunnerState& StateFor(std::uint32_t id);
  /// Rank 1's own greedy choice from the state's current logits.
  [[nodiscard]] OwnChoice ChooseGreedy(TextRunnerState& state) const;

  std::shared_ptr<TextModelRunner> runner_;
  std::vector<std::unique_ptr<TextRunnerState>> states_;
  std::uint64_t next_index_{0};
  std::uint64_t last_snapshot_id_{0};
  const std::size_t snapshot_budget_;
  std::size_t snapshot_bytes_{0};
  void DropSnapshot(std::uint64_t id);
  std::unordered_map<std::uint64_t, std::unique_ptr<TextRunnerSnapshot>>
      snapshots_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_TP_EXECUTOR_HPP_

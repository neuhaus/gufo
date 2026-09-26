#include "src/cli/serve/tp_executor.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace gufo::server {
namespace {

constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;
/// Recorded in place of a result when a call throws.
constexpr std::uint64_t kFailureMarker = 0xdead'c0de'0000'0001ULL;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

[[nodiscard]] std::string_view OpName(TpInstructionOp op) {
  switch (op) {
    case TpInstructionOp::kInvalidate:
      return "reset";
    case TpInstructionOp::kPrefill:
      return "prefill";
    case TpInstructionOp::kAdvance:
      return "advance";
    case TpInstructionOp::kDecode:
      return "decode";
    case TpInstructionOp::kSnapshot:
      return "snapshot";
    case TpInstructionOp::kDrop:
      return "drop";
    case TpInstructionOp::kRestore:
      return "restore";
    case TpInstructionOp::kReuse:
      return "reuse";
    case TpInstructionOp::kCancelPrepare:
      return "cancel-prepare";
    case TpInstructionOp::kEnd:
      return "end";
    case TpInstructionOp::kNone:
      break;
  }
  return "unknown";
}

[[nodiscard]] std::string Hex(std::uint64_t value) {
  char text[19];
  std::snprintf(text, sizeof(text), "%016llx",
                static_cast<unsigned long long>(value));
  return text;
}

[[nodiscard]] std::uint32_t ToWire(std::size_t value, const char* what) {
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error(std::string("TP instruction ") + what +
                              " exceeds the protocol range");
  }
  return static_cast<std::uint32_t>(value);
}

}  // namespace

void TpExecutionDigest::Add(std::uint64_t value) noexcept {
  for (unsigned shift = 0; shift < 64; shift += 8) {
    value_ ^= (value >> shift) & 0xffU;
    value_ *= kFnvPrime;
  }
}

void TpExecutionDigest::Add(std::span<const TextRunnerToken> tokens) noexcept {
  Add(tokens.size());
  for (const auto token : tokens) {
    Add(token);
  }
}

void DigestTpCall(TpExecutionDigest& digest, const TpInstruction& instruction,
                  std::span<const TextRunnerToken> prefill_prompt) {
  digest.Add(static_cast<std::uint64_t>(instruction.op));
  digest.Add(instruction.state);
  digest.Add(instruction.snapshot_id);
  digest.Add(static_cast<std::uint32_t>(instruction.token));
  digest.Add(instruction.offset);
  digest.Add(instruction.count);
  digest.Add(instruction.prompt_size);
  digest.Add(instruction.rng);
  digest.Add(static_cast<std::uint32_t>(instruction.pending));
  if (instruction.op == TpInstructionOp::kPrefill) {
    digest.Add(prefill_prompt);
  }
}

void DigestTpPrefill(TpExecutionDigest& digest, const TextPrefillStep& step,
                     std::size_t checkpoint) {
  digest.Add(step.consumed_tokens);
  digest.Add(step.decode_ready ? 1U : 0U);
  digest.Add(checkpoint);
}

void DigestTpAdvance(TpExecutionDigest& digest, std::size_t checkpoint) {
  digest.Add(checkpoint);
}

void DigestTpDecode(TpExecutionDigest& digest, const TextDecodeStep& step,
                    std::size_t checkpoint) {
  digest.Add(step.selections.size());
  for (const auto& selection : step.selections) {
    digest.Add(selection.token);
    digest.Add(selection.stop ? 1U : 0U);
  }
  digest.Add(step.stop ? 1U : 0U);
  digest.Add(step.draft_tokens);
  digest.Add(step.draft_accepted_tokens);
  digest.Add(checkpoint);
}

void DigestTpFailure(TpExecutionDigest& digest) {
  digest.Add(kFailureMarker);
}

bool TpCacheAcknowledged(TpInstructionOp op) noexcept {
  return op == TpInstructionOp::kSnapshot || op == TpInstructionOp::kRestore ||
         op == TpInstructionOp::kReuse || op == TpInstructionOp::kCancelPrepare;
}

void TpControlInstructionSink::Synchronize(std::uint64_t sequence,
                                           TpInstruction instruction,
                                           const std::function<void()>& local) {
  const std::lock_guard<std::mutex> lock(mutex_);
  instruction.index = next_index_;
  std::string error;
  if (!broker_->RegisterInstruction(sequence, instruction.index, &error)) {
    throw std::runtime_error(error);
  }
  if (!channel_->SendCommand({.sequence = sequence,
                              .kind = TpControlCommandKind::kInstruction,
                              .instruction = instruction},
                             &error)) {
    broker_->FailAll("TP cache instruction send failed: " + error);
    throw std::runtime_error(error);
  }
  ++next_index_;
  std::exception_ptr local_failure;
  try {
    local();
  } catch (...) {
    local_failure = std::current_exception();
  }
  // Drain even after a local failure: the next instruction must not overtake
  // a worker copy, and its acknowledgement must not become an orphan.
  const bool ok = broker_->WaitForInstruction(&error);
  if (local_failure)
    std::rethrow_exception(local_failure);
  if (!ok)
    throw std::runtime_error("TP cache worker failed: " + error);
}

bool TpControlInstructionSink::Send(std::uint64_t sequence,
                                    TpInstruction instruction,
                                    std::string* error) {
  const std::lock_guard<std::mutex> lock(mutex_);
  instruction.index = next_index_;
  const TpControlCommand command{
      .sequence = sequence,
      .kind = TpControlCommandKind::kInstruction,
      .instruction = instruction,
  };
  if (!channel_->SendCommand(command, error)) {
    return false;
  }
  ++next_index_;
  return true;
}

/// A rank-0 state paired with the id that names rank 1's corresponding state.
class TpMirroredRunner::State final : public TextRunnerState {
public:
  State(const TpMirroredRunner* owner, std::unique_ptr<TextRunnerState> inner,
        std::uint32_t id)
      : owner_(owner), inner_(std::move(inner)), id_(id) {}

  void Invalidate() noexcept override {
    owner_->SendInvalidate(id_);
    inner_->Invalidate();
  }
  /// Deliberately not forwarded; see `TpMirroredRunner`.
  void SetCancellationCheck(const CancellationCheck&) override {}
  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return inner_->MeasuredResources();
  }

  [[nodiscard]] TextRunnerState& inner() const noexcept { return *inner_; }
  [[nodiscard]] std::uint32_t id() const noexcept { return id_; }

private:
  const TpMirroredRunner* owner_;
  std::unique_ptr<TextRunnerState> inner_;
  std::uint32_t id_;
};

TpMirroredRunner::TpMirroredRunner(std::shared_ptr<TextModelRunner> inner,
                                   std::shared_ptr<TpInstructionSink> sink,
                                   std::size_t snapshot_budget)
    : inner_(std::move(inner)),
      sink_(std::move(sink)),
      snapshot_budget_(snapshot_budget) {
  if (inner_ == nullptr || sink_ == nullptr) {
    throw std::invalid_argument("TP mirrored runner needs a runner and a sink");
  }
  const auto capabilities = inner_->Descriptor().capabilities;
  multi_token_decode_ = capabilities.multi_token_decode;
}

void TpMirroredRunner::BeginRequest(std::uint64_t sequence) {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  const std::lock_guard<std::mutex> lock(mutex_);
  if (sequence == 0 || sequence_ != 0) {
    throw std::logic_error("TP request mirroring is already active or unnamed");
  }
  sequence_ = sequence;
  count_ = 0;
  digest_ = {};
}

bool TpMirroredRunner::EndRequest(std::string* error) {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto sequence = std::exchange(sequence_, 0);
  if (sequence == 0) {
    SetError(error, "no TP request is being mirrored");
    return false;
  }
  if (!failure_.empty()) {
    SetError(error, "TP instruction channel failed: " + failure_);
    return false;
  }
  if (count_ > std::numeric_limits<std::uint32_t>::max()) {
    SetError(error, "TP request made too many model calls to report");
    return false;
  }
  const TpInstruction end{.op = TpInstructionOp::kEnd,
                          .count = static_cast<std::uint32_t>(count_),
                          .digest = digest_.value()};
  std::string send_error;
  if (!sink_->Send(sequence, end, &send_error)) {
    failure_ = send_error.empty() ? "send failed" : send_error;
    SetError(error, "TP request end send failed: " + failure_);
    return false;
  }
  return true;
}

TpMirroredRunner::State& TpMirroredRunner::Mirrored(TextRunnerState& state) {
  auto* mirrored = dynamic_cast<State*>(&state);
  if (mirrored == nullptr) {
    throw std::logic_error("text runner state is not a TP2 mirrored state");
  }
  return *mirrored;
}

const TpMirroredRunner::State& TpMirroredRunner::Mirrored(
    const TextRunnerState& state) {
  const auto* mirrored = dynamic_cast<const State*>(&state);
  if (mirrored == nullptr) {
    throw std::logic_error("text runner state is not a TP2 mirrored state");
  }
  return *mirrored;
}

void TpMirroredRunner::Send(
    const TpInstruction& instruction,
    std::span<const TextRunnerToken> prefill_prompt) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!failure_.empty()) {
    throw std::runtime_error("TP instruction channel failed: " + failure_);
  }
  if (sequence_ == 0) {
    throw std::logic_error("TP2 model call outside a request");
  }
  DigestTpCall(digest_, instruction, prefill_prompt);
  ++count_;
  std::string error;
  if (!sink_->Send(sequence_, instruction, &error)) {
    failure_ = error.empty() ? "send failed" : error;
    throw std::runtime_error("TP instruction send failed: " + failure_);
  }
}

void TpMirroredRunner::SendInvalidate(std::uint32_t state) const noexcept {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
  try {
    lock.lock();
    if (!failure_.empty()) {
      return;
    }
    const TpInstruction instruction{.op = TpInstructionOp::kInvalidate,
                                    .state = state};
    if (sequence_ != 0) {
      DigestTpCall(digest_, instruction);
      ++count_;
    }
    std::string error;
    if (!sink_->Send(sequence_, instruction, &error)) {
      failure_ = error.empty() ? "reset send failed" : error;
    }
  } catch (...) {
    if (lock.owns_lock()) {
      try {
        failure_ = "reset send failed";
      } catch (...) {
      }
    }
  }
}

void TpMirroredRunner::Record(
    const std::function<void(TpExecutionDigest&)>& add) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  add(digest_);
}

TextRunnerDescriptor TpMirroredRunner::Descriptor() const {
  auto descriptor = inner_->Descriptor();
  auto& capabilities = descriptor.capabilities;
  capabilities.batched_multi_token_decode = false;
  capabilities.batched_multi_token_decode_max_width = 0;
  descriptor.persistence.reset();
  return descriptor;
}

TextRunnerResourceClaim TpMirroredRunner::ResourceClaim() const {
  auto claim = inner_->ResourceClaim();
  claim.retained_snapshot_capacity_bytes = std::min(
      claim.retained_snapshot_capacity_bytes.value_or(0), snapshot_budget_);
  return claim;
}

std::vector<TextExecutionPlan> TpMirroredRunner::SupportedPlans() const {
  return {{.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
}

std::vector<TextRunnerToken> TpMirroredRunner::Tokenize(
    std::string_view text) const {
  return inner_->Tokenize(text);
}

std::optional<std::vector<TextRunnerToken>> TpMirroredRunner::RenderAndTokenize(
    const ChatRequest& request) const {
  return inner_->RenderAndTokenize(request);
}

std::optional<TextPreparedPrompt> TpMirroredRunner::PreparePrompt(
    const ChatRequest& request) const {
  return inner_->PreparePrompt(request);
}

void TpMirroredRunner::SetPromptContext(
    TextRunnerState& state,
    std::shared_ptr<const TextPromptContext> context) const {
  if (context != nullptr) {
    throw std::invalid_argument("TP2 does not support prompt contexts");
  }
  inner_->SetPromptContext(Mirrored(state).inner(), std::move(context));
}

TextGenerationBackend::InitialOutputState TpMirroredRunner::InitialOutputState(
    const ChatRequest& request) const {
  return inner_->InitialOutputState(request);
}

std::string TpMirroredRunner::Decode(
    std::span<const TextRunnerToken> tokens) const {
  return inner_->Decode(tokens);
}

std::unique_ptr<TextRunnerState> TpMirroredRunner::CreateState() const {
  auto inner = inner_->CreateState();
  if (inner == nullptr) {
    return nullptr;
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  return std::make_unique<State>(this, std::move(inner), next_state_id_++);
}

std::optional<TextDecodeSelection> TpMirroredRunner::PreviewFirstToken(
    TextRunnerState& state, sampling::SamplerState& sampler) const {
  return inner_->PreviewFirstToken(Mirrored(state).inner(), sampler);
}

void TpMirroredRunner::PreparePrefixReuse(
    TextRunnerState& state, std::span<const TextRunnerToken> prefix) const {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  auto& mirrored = Mirrored(state);
  CacheCall({.op = TpInstructionOp::kReuse,
             .state = mirrored.id(),
             .prompt_size = ToWire(prefix.size(), "reuse prefix")},
            [&] { inner_->PreparePrefixReuse(mirrored.inner(), prefix); });
  Record([&](TpExecutionDigest& digest) {
    digest.Add(inner_->CheckpointPosition(mirrored.inner()));
  });
}

void TpMirroredRunner::PrepareBatchExecution(TextRunnerState& state) const {
  inner_->PrepareBatchExecution(Mirrored(state).inner());
}

TextPrefillStep TpMirroredRunner::Prefill(
    TextRunnerState& state, std::span<const TextRunnerToken> prompt,
    std::size_t offset, std::size_t max_input_tokens) const {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  auto& mirrored = Mirrored(state);
  if (offset >= prompt.size() || max_input_tokens == 0) {
    // Nothing to prefill: runners reject the call before any model work, so
    // there is nothing for rank 1 to join.
    return inner_->Prefill(mirrored.inner(), prompt, offset, max_input_tokens);
  }
  // A budget beyond the remaining prompt means the same as the remaining
  // prompt, and fits the wire; both ranks prefill with the clamped value.
  const std::size_t count = std::min(max_input_tokens, prompt.size() - offset);
  const TpInstruction instruction{
      .op = TpInstructionOp::kPrefill,
      .state = mirrored.id(),
      .offset = ToWire(offset, "offset"),
      .count = ToWire(count, "budget"),
      .prompt_size = ToWire(prompt.size(), "prompt")};
  Send(instruction, prompt);
  TextPrefillStep step;
  try {
    step = inner_->Prefill(mirrored.inner(), prompt, offset, count);
  } catch (...) {
    Record(DigestTpFailure);
    throw;
  }
  Record([&](TpExecutionDigest& digest) {
    DigestTpPrefill(digest, step, inner_->CheckpointPosition(mirrored.inner()));
  });
  return step;
}

TextDecodeSelection TpMirroredRunner::SelectNext(
    TextRunnerState& state, sampling::SamplerState& sampler) const {
  return inner_->SelectNext(Mirrored(state).inner(), sampler);
}

void TpMirroredRunner::Advance(TextRunnerState& state,
                               TextRunnerToken token) const {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  auto& mirrored = Mirrored(state);
  if (token >
      static_cast<TextRunnerToken>(std::numeric_limits<std::int32_t>::max())) {
    throw std::invalid_argument("TP2 token exceeds the instruction range");
  }
  Send({.op = TpInstructionOp::kAdvance,
        .state = mirrored.id(),
        .token = static_cast<std::int32_t>(token)});
  try {
    inner_->Advance(mirrored.inner(), token);
  } catch (...) {
    Record(DigestTpFailure);
    throw;
  }
  Record([&](TpExecutionDigest& digest) {
    DigestTpAdvance(digest, inner_->CheckpointPosition(mirrored.inner()));
  });
}

TextDecodeStep TpMirroredRunner::DecodeStep(
    TextRunnerState& state, std::size_t max_tokens,
    sampling::SamplerState& sampler) const {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  // A multi-token cycle is one instruction. It carries the sampler's draw
  // state, so rank 1, whose sampler otherwise matches, draws what rank 0 draws
  // and makes the same draft and acceptance decisions. A one-token budget runs
  // through the base implementation instead, which selects here, on rank 0
  // only, and mirrors the `Advance`; forwarding it to the wrapped runner would
  // let that `Advance` bypass this wrapper.
  if (max_tokens < 2 || !multi_token_decode_) {
    return TextModelRunner::DecodeStep(state, max_tokens, sampler);
  }
  auto& mirrored = Mirrored(state);
  const auto count = ToWire(max_tokens, "decode budget");
  const auto draw = sampler.SaveDrawState();
  if (draw.pending.has_value() &&
      *draw.pending > static_cast<sampling::TokenId>(
                          std::numeric_limits<std::int32_t>::max())) {
    throw std::invalid_argument("TP2 pending draw exceeds the protocol range");
  }
  Send({.op = TpInstructionOp::kDecode,
        .state = mirrored.id(),
        .count = count,
        .rng = draw.rng,
        .pending = draw.pending.has_value()
                       ? static_cast<std::int32_t>(*draw.pending)
                       : -1});
  TextDecodeStep step;
  try {
    step = inner_->DecodeStep(mirrored.inner(), count, sampler);
  } catch (...) {
    Record(DigestTpFailure);
    throw;
  }
  Record([&](TpExecutionDigest& digest) {
    DigestTpDecode(digest, step, inner_->CheckpointPosition(mirrored.inner()));
  });
  return step;
}

std::vector<TextDecodeStep> TpMirroredRunner::DecodeBatch(
    std::span<const TextRunnerDecode> decodes) const {
  if (decodes.size() > 1) {
    throw std::invalid_argument("TP2 does not mirror batched decoding yet");
  }
  return TextModelRunner::DecodeBatch(decodes);
}

void TpMirroredRunner::AdvanceBatch(
    std::span<const TextRunnerAdvance> advances) const {
  if (advances.size() > 1) {
    throw std::invalid_argument("TP2 does not mirror batched decoding yet");
  }
  TextModelRunner::AdvanceBatch(advances);
}

std::size_t TpMirroredRunner::CheckpointPosition(
    const TextRunnerState& state) const {
  return inner_->CheckpointPosition(Mirrored(state).inner());
}

void TpMirroredRunner::PrepareCancellation(TextRunnerState& state) const {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  auto& mirrored = Mirrored(state);
  CacheCall({.op = TpInstructionOp::kCancelPrepare, .state = mirrored.id()},
            [&] { inner_->PrepareCancellation(mirrored.inner()); });
}

class TpMirroredRunner::SnapshotHandle final : public TextRunnerSnapshot {
public:
  SnapshotHandle(std::shared_ptr<const TpMirroredRunner> owner,
                 std::unique_ptr<TextRunnerSnapshot> inner, std::uint64_t id)
      : owner(std::move(owner)), inner(std::move(inner)), id(id) {}
  ~SnapshotHandle() override { owner->Drop(id); }
  std::size_t PayloadBytes() const noexcept override {
    return inner->PayloadBytes();
  }
  std::shared_ptr<const TpMirroredRunner> owner;
  std::unique_ptr<TextRunnerSnapshot> inner;
  std::uint64_t id;
};

void TpMirroredRunner::CacheCall(const TpInstruction& instruction,
                                 const std::function<void()>& local) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (sequence_ == 0 || !failure_.empty()) {
    throw std::logic_error("TP cache call outside a healthy request");
  }
  DigestTpCall(digest_, instruction);
  ++count_;
  try {
    sink_->Synchronize(sequence_, instruction, local);
  } catch (...) {
    // A failed capture is a skipped snapshot on both ranks, not a failed call.
    if (instruction.op != TpInstructionOp::kSnapshot)
      DigestTpFailure(digest_);
    throw;
  }
}

void TpMirroredRunner::Drop(std::uint64_t id) const noexcept {
  try {
    const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!failure_.empty())
      return;
    const TpInstruction instruction{.op = TpInstructionOp::kDrop,
                                    .snapshot_id = id};
    if (sequence_ != 0) {
      DigestTpCall(digest_, instruction);
      ++count_;
    }
    std::string error;
    if (!sink_->Send(sequence_, instruction, &error))
      failure_ = "snapshot drop failed: " + error;
  } catch (...) {
    // Match reset's deferred channel failure behavior; destructors must not
    // throw.
    try {
      const std::lock_guard<std::mutex> lock(mutex_);
      failure_ = "snapshot drop failed";
    } catch (...) {
    }
  }
}

std::size_t TpMirroredRunner::SnapshotPayloadBytes(
    const TextRunnerState& state) const {
  return inner_->SnapshotPayloadBytes(Mirrored(state).inner());
}

std::unique_ptr<TextRunnerSnapshot> TpMirroredRunner::Snapshot(
    const TextRunnerState& state) const {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  const auto& mirrored = Mirrored(state);
  if (next_snapshot_id_ == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("TP snapshot ID exhausted");
  }
  const auto id = next_snapshot_id_++;
  std::unique_ptr<TextRunnerSnapshot> snapshot;
  try {
    CacheCall({.op = TpInstructionOp::kSnapshot,
               .state = mirrored.id(),
               .snapshot_id = id},
              [&] {
                snapshot = inner_->Snapshot(mirrored.inner());
                if (!snapshot)
                  throw std::runtime_error("snapshot capture returned null");
              });
    return std::make_unique<SnapshotHandle>(shared_from_this(),
                                            std::move(snapshot), id);
  } catch (...) {
    Drop(id);
    throw;
  }
}

void TpMirroredRunner::RestoreOrFork(TextRunnerState& state,
                                     const TextRunnerSnapshot& snapshot) const {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  const auto* handle = dynamic_cast<const SnapshotHandle*>(&snapshot);
  if (!handle || handle->owner.get() != this)
    throw std::invalid_argument("foreign TP snapshot");
  auto& mirrored = Mirrored(state);
  CacheCall({.op = TpInstructionOp::kRestore,
             .state = mirrored.id(),
             .snapshot_id = handle->id},
            [&] { inner_->RestoreOrFork(mirrored.inner(), *handle->inner); });
  Record([&](TpExecutionDigest& digest) {
    digest.Add(inner_->CheckpointPosition(mirrored.inner()));
  });
}

TpExecutor::TpExecutor(std::shared_ptr<TextModelRunner> runner,
                       std::size_t state_count, std::size_t snapshot_budget)
    : runner_(std::move(runner)), snapshot_budget_(snapshot_budget) {
  if (runner_ == nullptr || state_count == 0) {
    throw std::invalid_argument("TP executor needs a runner and a state");
  }
  states_.reserve(state_count);
  for (std::size_t index = 0; index < state_count; ++index) {
    auto state = runner_->CreateState();
    if (state == nullptr) {
      throw std::runtime_error("TP executor state creation failed");
    }
    states_.push_back(std::move(state));
  }
}

bool TpExecutor::TakeIndex(const TpControlCommand& command,
                           std::string* error) {
  if (command.instruction.index != next_index_) {
    SetError(error, "TP instruction " +
                        std::to_string(command.instruction.index) +
                        " arrived, expected " + std::to_string(next_index_));
    return false;
  }
  ++next_index_;
  return true;
}

TextRunnerState& TpExecutor::StateFor(std::uint32_t id) {
  if (id >= states_.size()) {
    throw std::out_of_range("TP instruction names state " + std::to_string(id) +
                            " of " + std::to_string(states_.size()));
  }
  return *states_[id];
}

TpExecutor::OwnChoice TpExecutor::ChooseGreedy(TextRunnerState& state) const {
  sampling::SamplerState greedy{sampling::SamplingConfig{}};
  const auto selection = runner_->SelectNext(state, greedy);
  return {.stop = selection.stop, .token = selection.token};
}

void TpExecutor::DropSnapshot(std::uint64_t id) {
  if (id == 0 || id > last_snapshot_id_)
    throw std::logic_error("unknown snapshot drop ID");
  const auto found = snapshots_.find(id);
  if (found != snapshots_.end()) {
    snapshot_bytes_ -= found->second->PayloadBytes();
    snapshots_.erase(found);
  }
}

bool TpExecutor::ExecuteIdle(const TpControlCommand& command,
                             std::string* error) {
  if (command.kind != TpControlCommandKind::kInstruction ||
      command.sequence != 0 ||
      (command.instruction.op != TpInstructionOp::kInvalidate &&
       command.instruction.op != TpInstructionOp::kDrop)) {
    SetError(error, "TP worker received a model call outside a request");
    return false;
  }
  if (!TakeIndex(command, error)) {
    return false;
  }
  try {
    if (command.instruction.op == TpInstructionOp::kDrop) {
      DropSnapshot(command.instruction.snapshot_id);
    } else {
      StateFor(command.instruction.state).Invalidate();
    }
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
  return true;
}

bool TpExecutor::RunRequest(const TpControlCommand& begin,
                            const Receive& receive,
                            const Acknowledge& acknowledge,
                            std::string* outcome, std::string* error) {
  std::vector<TextRunnerToken> prompt;
  prompt.reserve(begin.prompt_tokens.size());
  for (const auto token : begin.prompt_tokens) {
    if (token < 0) {
      SetError(error, "TP request prompt holds a negative token");
      return false;
    }
    prompt.push_back(static_cast<TextRunnerToken>(token));
  }
  // Under greedy decoding the logits are bit-identical on both ranks, so rank
  // 1's own choice must equal every token rank 0 advances. Rank 0's token is
  // what both ranks feed, so without this check a numerical divergence (the
  // full-Q8 bug was one) would go unnoticed. The choice is made after each call
  // while rank 0 is still selecting, so it costs rank 1 idle time only.
  const bool greedy = begin.sampling.can_use_unmodified_argmax();
  // Built as rank 0's pool builds the request's sampler, and fed the same
  // accepted tokens, so a multi-token decode sees the same history; each
  // `decode` instruction supplies rank 0's draw state.
  sampling::SamplerState sampler{begin.sampling, prompt};
  std::vector<std::optional<OwnChoice>> own(states_.size());
  TpExecutionDigest digest;
  std::uint64_t count = 0;
  std::string failure;
  const auto note = [&](std::string message) {
    if (failure.empty()) {
      failure = std::move(message);
    }
  };
  for (;;) {
    TpControlCommand command;
    if (!receive(&command, error)) {
      return false;
    }
    if (command.kind != TpControlCommandKind::kInstruction) {
      SetError(error, "TP worker expected an instruction for request " +
                          std::to_string(begin.sequence));
      return false;
    }
    if (!TakeIndex(command, error)) {
      return false;
    }
    if (command.sequence != begin.sequence) {
      SetError(error, "TP instruction names request " +
                          std::to_string(command.sequence) + " while request " +
                          std::to_string(begin.sequence) + " is open");
      return false;
    }
    const auto& instruction = command.instruction;
    if (instruction.op == TpInstructionOp::kEnd) {
      if (failure.empty() && (instruction.count != count ||
                              instruction.digest != digest.value())) {
        failure = "TP ranks ran different model calls: rank 0 made " +
                  std::to_string(instruction.count) + " with digest " +
                  Hex(instruction.digest) + ", rank 1 made " +
                  std::to_string(count) + " with digest " + Hex(digest.value());
      }
      *outcome = std::move(failure);
      return true;
    }
    ++count;
    const bool prompt_fits = instruction.prompt_size <= prompt.size();
    const auto prefill_prompt =
        instruction.op == TpInstructionOp::kPrefill && prompt_fits
            ? std::span<const TextRunnerToken>(prompt).first(
                  instruction.prompt_size)
            : std::span<const TextRunnerToken>{};
    DigestTpCall(digest, instruction, prefill_prompt);
    std::string call_error;
    try {
      auto& state = StateFor(instruction.state);
      auto& choice = own[instruction.state];
      switch (instruction.op) {
        case TpInstructionOp::kSnapshot: {
          if (instruction.snapshot_id <= last_snapshot_id_)
            throw std::logic_error("reused snapshot ID");
          last_snapshot_id_ = instruction.snapshot_id;
          const auto estimate = runner_->SnapshotPayloadBytes(state);
          if (estimate > snapshot_budget_ - snapshot_bytes_)
            throw std::runtime_error("worker snapshot budget exhausted");
          auto snapshot = runner_->Snapshot(state);
          if (!snapshot)
            throw std::runtime_error("snapshot capture returned null");
          const auto bytes = snapshot->PayloadBytes();
          if (bytes > snapshot_budget_ - snapshot_bytes_)
            throw std::runtime_error("worker snapshot exceeds reserved budget");
          snapshots_.emplace(instruction.snapshot_id, std::move(snapshot));
          snapshot_bytes_ += bytes;
          break;
        }
        case TpInstructionOp::kDrop:
          DropSnapshot(instruction.snapshot_id);
          break;
        case TpInstructionOp::kRestore: {
          choice.reset();
          const auto found = snapshots_.find(instruction.snapshot_id);
          if (found == snapshots_.end())
            throw std::logic_error("unknown snapshot ID");
          runner_->RestoreOrFork(state, *found->second);
          digest.Add(runner_->CheckpointPosition(state));
          break;
        }
        case TpInstructionOp::kReuse:
          choice.reset();
          if (!prompt_fits)
            throw std::invalid_argument("reuse exceeds request prompt");
          runner_->PreparePrefixReuse(
              state, std::span<const TextRunnerToken>(prompt).first(
                         instruction.prompt_size));
          digest.Add(runner_->CheckpointPosition(state));
          if (greedy)
            choice = ChooseGreedy(state);
          break;
        case TpInstructionOp::kCancelPrepare:
          runner_->PrepareCancellation(state);
          break;
        case TpInstructionOp::kInvalidate:
          choice.reset();
          state.Invalidate();
          break;
        case TpInstructionOp::kPrefill: {
          if (!prompt_fits) {
            throw std::invalid_argument("prefill exceeds the request prompt");
          }
          choice.reset();
          const auto step = runner_->Prefill(
              state, prefill_prompt, instruction.offset, instruction.count);
          DigestTpPrefill(digest, step, runner_->CheckpointPosition(state));
          if (greedy && step.decode_ready) {
            choice = ChooseGreedy(state);
          }
          break;
        }
        case TpInstructionOp::kAdvance: {
          const auto token = static_cast<TextRunnerToken>(instruction.token);
          if (greedy &&
              (!choice.has_value() || choice->stop || choice->token != token)) {
            note("rank 1 " +
                 (choice.has_value()
                      ? (choice->stop ? std::string("stopped")
                                      : "selected token " +
                                            std::to_string(choice->token))
                      : std::string("had no token")) +
                 " where rank 0 advanced token " + std::to_string(token) +
                 " at instruction " + std::to_string(instruction.index));
          }
          choice.reset();
          runner_->Advance(state, token);
          sampler.Accept(token);
          DigestTpAdvance(digest, runner_->CheckpointPosition(state));
          if (greedy) {
            choice = ChooseGreedy(state);
          }
          break;
        }
        case TpInstructionOp::kDecode: {
          choice.reset();
          sampler.RestoreDrawState(
              {.rng = instruction.rng,
               .pending = instruction.pending >= 0
                              ? std::optional<sampling::TokenId>(
                                    static_cast<sampling::TokenId>(
                                        instruction.pending))
                              : std::nullopt});
          const auto step =
              runner_->DecodeStep(state, instruction.count, sampler);
          for (const auto& selection : step.selections) {
            sampler.Accept(selection.token);
          }
          DigestTpDecode(digest, step, runner_->CheckpointPosition(state));
          if (greedy) {
            choice = ChooseGreedy(state);
          }
          break;
        }
        case TpInstructionOp::kEnd:
        case TpInstructionOp::kNone:
          throw std::logic_error("TP instruction has no model call");
      }
    } catch (const std::exception& exception) {
      call_error = "rank 1 " + std::string(OpName(instruction.op)) +
                   " failed: " + exception.what();
      // A failed capture only means no snapshot: the acknowledgement makes
      // rank 0 skip it too and drop the ID, and neither state changed.
      if (instruction.op != TpInstructionOp::kSnapshot) {
        DigestTpFailure(digest);
        note(call_error);
      }
    }
    if (TpCacheAcknowledged(instruction.op)) {
      const TpControlResponse response{
          .instruction_index = instruction.index,
          .sequence = begin.sequence,
          .error = call_error,
          .kind = TpControlResponseKind::kInstruction};
      if (!acknowledge(response, error))
        return false;
    }
  }
}

}  // namespace gufo::server

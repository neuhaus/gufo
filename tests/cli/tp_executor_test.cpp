// Drives the real scheduler and runner pool on rank 0 through
// `TpMirroredRunner`, and a `TpExecutor` on rank 1, over a loopback control
// channel. A toy model records every model call each rank makes; the pair is
// correct when both ranks record the same calls with the same results, for
// greedy, sampled, stopped, cancelled and multi-token requests, and when rank 1
// reports the divergences it is meant to catch.

#include "src/cli/serve/tp_executor.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/cli/serve/tp_control.hpp"

namespace {

using gufo::server::ChatRequest;
using gufo::server::TextDecodeSelection;
using gufo::server::TextDecodeStep;
using gufo::server::TextExecutionPlan;
using gufo::server::TextExecutionPlanKind;
using gufo::server::TextGenerationBackend;
using gufo::server::TextGenerationScheduler;
using gufo::server::TextModelRunner;
using gufo::server::TextPrefillStep;
using gufo::server::TextRequestMetadata;
using gufo::server::TextRunnerCapabilities;
using gufo::server::TextRunnerDescriptor;
using gufo::server::TextRunnerPool;
using gufo::server::TextRunnerResourceClaim;
using gufo::server::TextRunnerState;
using gufo::server::TextRunnerToken;
using gufo::server::TpControlChannel;
using gufo::server::TpControlCommand;
using gufo::server::TpControlCommandKind;
using gufo::server::TpControlConfig;
using gufo::server::TpControlInstructionSink;
using gufo::server::TpControlResponse;
using gufo::server::TpExecutor;
using gufo::server::TpInstructionOp;
using gufo::server::TpMirroredRunner;
using FinishReason = TextGenerationBackend::FinishReason;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    std::exit(1);
  }
}

std::uint16_t FreePort() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  ::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address));
  socklen_t length = sizeof(address);
  ::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length);
  const auto port = ntohs(address.sin_port);
  ::close(fd);
  return port;
}

constexpr std::size_t kVocab = 32;
constexpr TextRunnerToken kEos = 0;

/// One model call and what it produced, as a comparable line.
using CallLog = std::vector<std::string>;

struct ToyOptions {
  bool mtp{false};
  std::size_t prefill_chunk{3};
  /// Rank-1 faults: shift the greedy choice once, at this sequence length.
  std::optional<std::size_t> diverge_at;
  /// Rank-1 fault: report one extra draft in the next multi-token step.
  bool extra_draft_once{false};
};

class ToyRunner;

class ToyState final : public TextRunnerState {
public:
  ToyState(const ToyRunner* owner, std::size_t id) : owner_(owner), id_(id) {}
  void Invalidate() noexcept override;
  void SetCancellationCheck(const CancellationCheck&) override;

  std::vector<TextRunnerToken> tokens;
  std::size_t id() const { return id_; }

private:
  const ToyRunner* owner_;
  std::size_t id_;
};

/// A deterministic model whose next token depends on the whole sequence, so
/// any divergence between the ranks changes later tokens too.
class ToyRunner final : public TextModelRunner {
public:
  explicit ToyRunner(ToyOptions options) : options_(options) {}

  CallLog Log() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return log_;
  }
  void Record(std::string line) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    log_.push_back(std::move(line));
  }
  std::size_t prefill_calls() const { return prefill_calls_.load(); }
  std::size_t cancellation_checks() const { return cancellation_checks_; }
  void CountCancellationCheck() const { ++cancellation_checks_; }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {.model_id = "toy",
            .state_abi = "toy-v1",
            .max_context = 256,
            .capabilities = TextRunnerCapabilities{
                .incremental_prefill = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = options_.mtp,
                .prefix_reuse = true,
            }};
  }
  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    return {.resident_weights_bytes = 0,
            .state_capacity_bytes = 8 * 64,
            .per_request_state_bytes = 64,
            .temporary_scratch_bytes = 0,
            .retained_snapshot_capacity_bytes = 0,
            .requires_device_runtime_lock = true};
  }
  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    return {{.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
  }
  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return {static_cast<TextRunnerToken>(text.size())};
  }
  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest&) const override {
    return std::nullopt;
  }
  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    std::string text;
    for (const auto token : tokens) {
      text += Piece(token);
    }
    return text;
  }
  static std::string Piece(TextRunnerToken token) {
    return " t" + std::to_string(token) + ";";
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<ToyState>(this, next_state_++);
  }
  void PreparePrefixReuse(TextRunnerState&,
                          std::span<const TextRunnerToken>) const override {
    Record("unexpected prefix reuse");
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& toy = Toy(state);
    if (offset != toy.tokens.size() || offset >= prompt.size()) {
      throw std::logic_error("toy prefill offset mismatch");
    }
    const auto consumed = std::min(
        {max_input_tokens, prompt.size() - offset, options_.prefill_chunk});
    toy.tokens.insert(toy.tokens.end(), prompt.begin() + offset,
                      prompt.begin() + offset + consumed);
    ++prefill_calls_;
    Record("prefill s" + std::to_string(toy.id()) + " @" +
           std::to_string(offset) + " +" + std::to_string(consumed) + "/" +
           std::to_string(prompt.size()));
    return {.consumed_tokens = consumed,
            .decode_ready = toy.tokens.size() == prompt.size()};
  }
  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state,
      gufo::sampling::SamplerState& sampler) const override {
    auto& toy = Toy(state);
    const auto logits = Logits(toy.tokens);
    const auto token = static_cast<TextRunnerToken>(sampler.Sample(logits));
    if (token == kEos) {
      return {.stop = true};
    }
    return {.stop = false, .token = token, .piece = Piece(token)};
  }
  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& toy = Toy(state);
    toy.tokens.push_back(token);
    Record("advance s" + std::to_string(toy.id()) + " " +
           std::to_string(token));
  }
  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      gufo::sampling::SamplerState& sampler) const override {
    if (!options_.mtp || max_tokens < 2) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    auto& toy = Toy(state);
    TextDecodeStep step;
    std::string produced;
    // Like Flash-Next: draw on a working copy, leave a deferred draw for the
    // next call when sampling at random, and publish the draw state back.
    gufo::sampling::SamplerState working = sampler;
    for (std::size_t i = 0; i < std::min<std::size_t>(max_tokens, 3); ++i) {
      const auto token =
          static_cast<TextRunnerToken>(working.Sample(Logits(toy.tokens)));
      if (token == kEos) {
        step.stop = true;
        break;
      }
      toy.tokens.push_back(token);
      step.selections.push_back({.token = token, .piece = Piece(token)});
      produced += " " + std::to_string(token);
    }
    if (!step.stop && sampler.config().uses_random_sampling()) {
      working.DeferSample(working.Sample(Logits(toy.tokens)));
      produced += " deferred";
    }
    sampler.CopyDrawStateFrom(working);
    step.draft_tokens = 3;
    step.draft_accepted_tokens =
        step.selections.empty() ? 0 : step.selections.size() - 1;
    if (options_.extra_draft_once) {
      options_.extra_draft_once = false;
      ++step.draft_tokens;
    }
    Record("decode s" + std::to_string(toy.id()) + " max" +
           std::to_string(max_tokens) + produced);
    return step;
  }
  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return dynamic_cast<const ToyState&>(state).tokens.size();
  }

private:
  static ToyState& Toy(TextRunnerState& state) {
    return dynamic_cast<ToyState&>(state);
  }
  std::vector<float> Logits(const std::vector<TextRunnerToken>& tokens) const {
    std::size_t sum = 0;
    for (const auto token : tokens) {
      sum += token;
    }
    // Never EOS unless the sequence is long, so budgets decide most lengths.
    std::size_t peak = (sum * 7 + tokens.size() * 3) % (kVocab - 1) + 1;
    if (tokens.size() >= 40) {
      peak = kEos;
    }
    if (options_.diverge_at == tokens.size()) {
      options_.diverge_at.reset();
      peak = peak % (kVocab - 1) + 1;
    }
    std::vector<float> logits(kVocab);
    for (std::size_t i = 0; i < kVocab; ++i) {
      logits[i] = -0.25F * static_cast<float>(i > peak ? i - peak : peak - i);
    }
    return logits;
  }

  mutable ToyOptions options_;
  mutable std::mutex mutex_;
  mutable CallLog log_;
  mutable std::size_t next_state_{0};
  mutable std::atomic<std::size_t> prefill_calls_{0};
  mutable std::atomic<std::size_t> cancellation_checks_{0};
};

void ToyState::Invalidate() noexcept {
  tokens.clear();
  try {
    owner_->Record("reset s" + std::to_string(id_));
  } catch (...) {
  }
}

void ToyState::SetCancellationCheck(const CancellationCheck&) {
  owner_->CountCancellationCheck();
}

/// Rank 0 with the real scheduler, rank 1 with an executor, and the channel
/// between them.
class Pair {
public:
  Pair(ToyOptions rank0_options, ToyOptions rank1_options)
      : inner0_(std::make_shared<ToyRunner>(rank0_options)),
        runner1_(std::make_shared<ToyRunner>(rank1_options)) {
    const auto port = FreePort();
    std::string server_error;
    std::string client_error;
    std::thread connector([&] {
      client_ = TpControlChannel::Connect("127.0.0.1", port, &client_error);
    });
    server_ = TpControlChannel::Listen(port, &server_error);
    connector.join();
    Require(server_ != nullptr && client_ != nullptr,
            "channel pair: " + server_error + client_error);
    const TpControlConfig rank0{.rank = 0,
                                .world_size = 2,
                                .max_context = 256,
                                .auth_token = "executor-test"};
    TpControlConfig rank1 = rank0;
    rank1.rank = 1;
    bool server_ok = false;
    bool client_ok = false;
    std::thread handshake(
        [&] { client_ok = client_->Handshake(rank1, &client_error); });
    server_ok = server_->Handshake(rank0, &server_error);
    handshake.join();
    Require(server_ok && client_ok,
            "handshake: " + server_error + client_error);

    sink_ = std::make_shared<TpControlInstructionSink>(server_);
    mirrored_ = std::make_shared<TpMirroredRunner>(inner0_, sink_);
    scheduler_ = std::make_shared<TextGenerationScheduler>(
        std::make_shared<TextRunnerPool>(mirrored_, 1));
    executor_ = std::make_unique<TpExecutor>(runner1_, 1);
    worker_ = std::thread([this] { Work(); });
  }

  ~Pair() {
    scheduler_.reset();
    server_.reset();
    client_->Interrupt();
    worker_.join();
  }

  struct Outcome {
    TextGenerationBackend::Result result;
    std::string worker_error;
  };

  Outcome Run(std::vector<TextRunnerToken> prompt, std::size_t max_tokens,
              gufo::sampling::SamplingConfig sampling = {},
              std::vector<std::string> stop_sequences = {},
              TextGenerationScheduler::CancellationCheck is_cancelled = {}) {
    const auto sequence = next_sequence_++;
    TpControlCommand begin{.sequence = sequence,
                           .max_tokens = static_cast<std::uint32_t>(max_tokens),
                           .client_id = "executor-test",
                           .sampling = sampling};
    for (const auto token : prompt) {
      begin.prompt_tokens.push_back(static_cast<std::int32_t>(token));
    }
    std::string error;
    Require(server_->SendCommand(begin, &error), "begin: " + error);
    mirrored_->BeginRequest(sequence);
    TextRequestMetadata metadata;
    metadata.cache_prompt = false;
    metadata.stop_sequences = std::move(stop_sequences);
    auto request = scheduler_->Submit(std::move(prompt), max_tokens, sampling,
                                      is_cancelled, false, std::move(metadata));
    Outcome outcome{.result = request.Wait(), .worker_error = {}};
    Require(mirrored_->EndRequest(&error), "end: " + error);
    TpControlResponse response;
    Require(server_->ReceiveResponse(&response, &error), "response: " + error);
    Require(response.sequence == sequence, "response sequence");
    outcome.worker_error = response.error;
    return outcome;
  }

  /// A reset sent between requests, as the continuation cache may send one.
  void ResetBetweenRequests() {
    std::string error;
    Require(sink_->Send(0, {.op = TpInstructionOp::kInvalidate, .state = 0},
                        &error),
            "idle reset: " + error);
  }

  void RequireSameCalls(const std::string& what) const {
    const auto rank0 = inner0_->Log();
    const auto rank1 = runner1_->Log();
    if (rank0 != rank1) {
      std::fprintf(stderr, "rank 0 calls:\n");
      for (const auto& line : rank0) {
        std::fprintf(stderr, "  %s\n", line.c_str());
      }
      std::fprintf(stderr, "rank 1 calls:\n");
      for (const auto& line : rank1) {
        std::fprintf(stderr, "  %s\n", line.c_str());
      }
    }
    Require(rank0 == rank1, what + ": both ranks make the same model calls");
    Require(worker_failure_.empty(), what + ": worker " + worker_failure_);
  }

  const ToyRunner& rank0() const { return *inner0_; }
  const ToyRunner& rank1() const { return *runner1_; }

private:
  void Work() {
    for (;;) {
      TpControlCommand command;
      std::string error;
      if (!client_->ReceiveCommand(&command, &error)) {
        return;
      }
      if (command.kind == TpControlCommandKind::kInstruction) {
        if (!executor_->ExecuteIdle(command, &error)) {
          worker_failure_ = error;
          return;
        }
        continue;
      }
      std::string outcome;
      if (!executor_->RunRequest(
              command,
              [this](TpControlCommand* next, std::string* receive_error) {
                return client_->ReceiveCommand(next, receive_error);
              },
              &outcome, &error)) {
        worker_failure_ = error;
        return;
      }
      const TpControlResponse response{.sequence = command.sequence,
                                       .error = outcome};
      if (!client_->SendResponse(response, &error)) {
        worker_failure_ = error;
        return;
      }
    }
  }

  std::shared_ptr<ToyRunner> inner0_;
  std::shared_ptr<ToyRunner> runner1_;
  std::shared_ptr<TpControlChannel> server_;
  std::shared_ptr<TpControlChannel> client_;
  std::shared_ptr<TpControlInstructionSink> sink_;
  std::shared_ptr<TpMirroredRunner> mirrored_;
  std::shared_ptr<TextGenerationScheduler> scheduler_;
  std::unique_ptr<TpExecutor> executor_;
  std::thread worker_;
  std::string worker_failure_;
  std::uint64_t next_sequence_{1};
};

std::size_t Count(const CallLog& log, const std::string& prefix) {
  return static_cast<std::size_t>(std::count_if(
      log.begin(), log.end(),
      [&](const std::string& line) { return line.rfind(prefix, 0) == 0; }));
}

gufo::sampling::SamplingConfig Sampled() {
  gufo::sampling::SamplingConfig config;
  config.temperature = 0.9F;
  config.seed = 42;
  // History matters only with a penalty, so this also checks that rank 1's
  // sampler accepts the same tokens as rank 0's.
  config.repeat_penalty = 1.3F;
  return config;
}

}  // namespace

int main() {
  const std::vector<TextRunnerToken> prompt{5, 9, 13, 17, 21, 3, 8};

  // AR: greedy, sampled, stopped and cancelled requests on one pair.
  {
    Pair pair({}, {});
    const auto greedy = pair.Run(prompt, 6);
    Require(greedy.worker_error.empty(), "greedy: " + greedy.worker_error);
    Require(greedy.result.tokens.size() == 6 &&
                greedy.result.finish_reason == FinishReason::kLength,
            "greedy request produces its budget");
    pair.RequireSameCalls("greedy");
    const auto calls = pair.rank0().Log();
    Require(Count(calls, "prefill") == 3,
            "a 7-token prompt prefills in three chunks of three");
    // The last token is published without advancing it.
    Require(Count(calls, "advance") == 5, "greedy advances all but the last");
    Require(pair.rank0().cancellation_checks() == 0,
            "the model session never receives a cancellation check");

    const auto again = pair.Run(prompt, 6);
    Require(again.worker_error.empty() &&
                again.result.tokens == greedy.result.tokens,
            "a repeated request is reproducible: " + again.worker_error);
    pair.RequireSameCalls("repeated");

    const auto sampled = pair.Run(prompt, 8, Sampled());
    Require(sampled.worker_error.empty(), "sampled: " + sampled.worker_error);
    Require(sampled.result.tokens.size() == 8, "sampled request completes");
    pair.RequireSameCalls("sampled");

    // Stop at a generated piece that does not occur earlier in the output.
    std::optional<std::size_t> stop_at;
    std::string text;
    for (std::size_t i = 0; i < greedy.result.tokens.size(); ++i) {
      const auto piece = ToyRunner::Piece(greedy.result.tokens[i]);
      if (i >= 2 && text.find(piece) == std::string::npos) {
        stop_at = i;
        break;
      }
      text += piece;
    }
    Require(stop_at.has_value(), "the greedy output has a usable stop piece");
    const auto stop_piece = ToyRunner::Piece(greedy.result.tokens[*stop_at]);
    const auto stopped = pair.Run(prompt, 6, {}, {stop_piece});
    Require(stopped.worker_error.empty(), "stop: " + stopped.worker_error);
    Require(stopped.result.finish_reason == FinishReason::kStopSequence &&
                stopped.result.stop_sequence == stop_piece,
            "the request ends at its stop sequence");
    pair.RequireSameCalls("stop sequence");

    // Cancel after the second prefill chunk of a long prompt.
    const auto before = pair.rank0().prefill_calls();
    const std::vector<TextRunnerToken> long_prompt{1, 2,  3,  4,  5,  6,  7, 8,
                                                   9, 10, 11, 12, 13, 14, 15};
    const auto cancelled = pair.Run(long_prompt, 6, {}, {}, [&pair, before] {
      return pair.rank0().prefill_calls() >= before + 2;
    });
    Require(cancelled.worker_error.empty(),
            "cancel: " + cancelled.worker_error);
    Require(cancelled.result.cancelled ||
                cancelled.result.finish_reason == FinishReason::kCancelled,
            "the request is cancelled");
    Require(pair.rank0().prefill_calls() < before + 5,
            "cancellation stops prefill at a chunk boundary");
    pair.RequireSameCalls("cancelled during prefill");

    // A reset between requests reaches rank 1 too.
    pair.ResetBetweenRequests();
    const auto after_reset = pair.Run(prompt, 3);
    Require(after_reset.worker_error.empty(),
            "after an idle reset: " + after_reset.worker_error);
    Require(Count(pair.rank1().Log(), "reset") ==
                Count(pair.rank0().Log(), "reset") + 1,
            "rank 1 executed the reset sent between requests");
  }

  // Multi-token decoding is one instruction per cycle, greedy or sampled; a
  // sampled cycle carries rank 0's draw state.
  {
    Pair pair({.mtp = true}, {.mtp = true});
    const auto greedy = pair.Run(prompt, 7);
    Require(greedy.worker_error.empty(), "MTP greedy: " + greedy.worker_error);
    Require(greedy.result.tokens.size() == 7, "MTP greedy completes");
    Require(Count(pair.rank0().Log(), "decode") > 0,
            "greedy MTP decodes in multi-token cycles");
    pair.RequireSameCalls("MTP greedy");

    const auto decodes = Count(pair.rank0().Log(), "decode");
    const auto sampled = pair.Run(prompt, 9, Sampled());
    Require(sampled.worker_error.empty(),
            "MTP sampled: " + sampled.worker_error);
    Require(sampled.result.tokens.size() == 9, "MTP sampled completes");
    Require(Count(pair.rank0().Log(), "decode") > decodes,
            "sampled MTP decodes in multi-token cycles too");
    pair.RequireSameCalls("MTP sampled");
    const auto again = pair.Run(prompt, 9, Sampled());
    Require(
        again.worker_error.empty() &&
            again.result.tokens == sampled.result.tokens,
        "a seeded sampled MTP request is reproducible: " + again.worker_error);
    pair.RequireSameCalls("MTP sampled again");

    // Without a seed each rank's sampler starts from its own random state, so
    // only rank 0's draw state, carried by each decode, keeps them together.
    auto unseeded = Sampled();
    unseeded.seed = -1;
    const auto random_draws = pair.Run(prompt, 9, unseeded);
    Require(random_draws.worker_error.empty(),
            "unseeded sampled MTP: " + random_draws.worker_error);
    pair.RequireSameCalls("MTP sampled without a seed");
  }

  // Rank 1's own greedy choice catches a numerical divergence.
  {
    Pair pair({}, {.diverge_at = prompt.size() + 2});
    const auto diverged = pair.Run(prompt, 6);
    Require(diverged.worker_error.find("selected token") != std::string::npos,
            "a rank-1 divergence fails the request: " + diverged.worker_error);
    // Rank 1 still advanced rank 0's tokens, so the pair stays usable.
    pair.RequireSameCalls("after divergence");
    const auto next = pair.Run(prompt, 6);
    Require(next.worker_error.empty(),
            "the next request succeeds: " + next.worker_error);
  }

  // The execution digest catches a result the ranks disagree on.
  {
    Pair pair({.mtp = true}, {.mtp = true, .extra_draft_once = true});
    const auto mismatch = pair.Run(prompt, 7);
    Require(mismatch.worker_error.find("different model calls") !=
                std::string::npos,
            "a result difference fails the request: " + mismatch.worker_error);
    const auto next = pair.Run(prompt, 7);
    Require(next.worker_error.empty(),
            "the next request succeeds: " + next.worker_error);
  }

  std::puts(
      "PASS: TP2 executor mirrors greedy, sampled, stopped, cancelled and "
      "multi-token requests and reports divergence");
  return 0;
}

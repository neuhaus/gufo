#ifndef GUFO_SERVER_TEXT_GENERATION_SCHEDULER_HPP_
#define GUFO_SERVER_TEXT_GENERATION_SCHEDULER_HPP_

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/serve/text_model_runner.hpp"

namespace gufo::server {

enum class TextRequestPhase : std::uint8_t {
  kQueued,
  kAdmitted,
  kPrefilling,
  kDecodeReady,
  kDecoding,
  kTerminal,
};

inline constexpr std::size_t kDefaultDecodeActivePrefillTokens = 512;
inline constexpr std::size_t kDefaultMaxOutputBytes =
    static_cast<std::size_t>(1024) * 1024;
inline constexpr std::size_t kDefaultMaxBufferedOutputBytes =
    static_cast<std::size_t>(64) * 1024;
inline constexpr std::size_t kDefaultMaxBufferedOutputBytesTotal =
    static_cast<std::size_t>(256) * 1024;
/// The first C2 slice is fixed at two members.
inline constexpr std::size_t kCohort2MemberCount = 2;

struct TextPrefillPolicy {
  std::size_t decode_active_tokens{kDefaultDecodeActivePrefillTokens};
};

struct TextSchedulerPolicy {
  std::size_t max_pending_requests{16};
  std::size_t max_pending_requests_per_client{4};
  std::size_t max_output_bytes_per_request{kDefaultMaxOutputBytes};
  std::size_t max_buffered_output_bytes_per_request{
      kDefaultMaxBufferedOutputBytes};
  std::size_t max_buffered_output_bytes_total{
      kDefaultMaxBufferedOutputBytesTotal};
  std::chrono::milliseconds request_timeout{0};
};

/// Single-owner scheduler for opaque text-model runner states.
///
/// Submitters never execute model code. One scheduler thread owns admission,
/// runner leases, prefill/decode work units, cancellation, and reclamation.
class TextGenerationScheduler {
public:
  using Clock = std::chrono::steady_clock;
  using Result = TextGenerationBackend::Result;
  using CancellationCheck = TextGenerationBackend::CancellationCheck;
  using TokenCallback = TextGenerationBackend::TokenCallback;

  struct RequestMetadata {
    std::string client_id{"anonymous"};
    std::optional<Clock::time_point> deadline;
    Clock::time_point request_start{Clock::now()};
    std::shared_ptr<const TextPromptContext> prompt_context;
    bool cache_prompt{true};
    std::size_t cache_prefix_tokens{0};
    std::vector<std::string> stop_sequences;
  };

  class Request {
  public:
    Request();
    ~Request();

    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;
    Request(Request&&) noexcept;
    Request& operator=(Request&&) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] std::uint64_t id() const noexcept;
    [[nodiscard]] TextRequestPhase phase() const noexcept;

    /// Consumes queued output pieces on the calling thread and waits for the
    /// scheduler-owned request to become terminal.
    Result Wait(const TokenCallback& on_token = {});
    void Cancel() noexcept;

  private:
    friend class TextGenerationScheduler;
    struct Impl;

    explicit Request(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
  };

  /// One externally identified member of a fixed two-member C2 cohort.
  /// This record describes scheduler admission only; it carries no distributed
  /// execution plan and never selects a shared C2 collective.
  struct CohortMemberRequest {
    std::uint64_t member_id{0};
    std::vector<TextRunnerToken> prompt;
    std::size_t max_tokens{1};
    sampling::SamplingConfig sampling;
    CancellationCheck is_cancelled;
    bool publish_token_pieces{false};
    RequestMetadata metadata;
  };

  /// Move-only handle for one atomically admitted, ordered C2 cohort.
  class Cohort {
  public:
    Cohort(const Cohort&) = delete;
    Cohort& operator=(const Cohort&) = delete;
    Cohort(Cohort&&) noexcept = default;
    Cohort& operator=(Cohort&&) noexcept = default;
    ~Cohort() = default;

    [[nodiscard]] std::uint64_t id() const noexcept { return id_; }
    [[nodiscard]] const std::array<std::uint64_t, kCohort2MemberCount>&
    member_ids() const noexcept {
      return member_ids_;
    }
    [[nodiscard]] const std::array<Request, kCohort2MemberCount>& members()
        const noexcept {
      return members_;
    }
    [[nodiscard]] std::array<Request, kCohort2MemberCount>& members() noexcept {
      return members_;
    }

  private:
    friend class TextGenerationScheduler;

    Cohort(std::uint64_t id,
           std::array<std::uint64_t, kCohort2MemberCount> member_ids,
           std::array<Request, kCohort2MemberCount> members);

    std::uint64_t id_{0};
    std::array<std::uint64_t, kCohort2MemberCount> member_ids_{};
    std::array<Request, kCohort2MemberCount> members_{};
  };

  explicit TextGenerationScheduler(std::shared_ptr<TextRunnerPool> runner_pool,
                                   TextPrefillPolicy prefill_policy = {},
                                   TextSchedulerPolicy scheduler_policy = {});
  ~TextGenerationScheduler();

  TextGenerationScheduler(const TextGenerationScheduler&) = delete;
  TextGenerationScheduler& operator=(const TextGenerationScheduler&) = delete;
  TextGenerationScheduler(TextGenerationScheduler&&) = delete;
  TextGenerationScheduler& operator=(TextGenerationScheduler&&) = delete;

  [[nodiscard]] const TextModelRunner& runner() const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t buffered_output_bytes() const noexcept;
  [[nodiscard]] std::size_t max_buffered_output_bytes() const noexcept;

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens,
                               const sampling::SamplingConfig& sampling,
                               const CancellationCheck& is_cancelled = {},
                               bool publish_token_pieces = false);

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens,
                               const sampling::SamplingConfig& sampling,
                               const CancellationCheck& is_cancelled,
                               bool publish_token_pieces,
                               RequestMetadata metadata);

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens, float temperature,
                               const CancellationCheck& is_cancelled = {},
                               bool publish_token_pieces = false);

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens, float temperature,
                               const CancellationCheck& is_cancelled,
                               bool publish_token_pieces,
                               RequestMetadata metadata);

  /// Admits one fixed two-member C2 cohort as a single ordered unit.
  ///
  /// Exactly two members with nonzero, unique IDs are required. An incomplete
  /// cohort, a duplicate ID, or a third-member join is rejected before either
  /// member can reach a runner. Members must be greedy, non-streaming, uncached
  /// requests without prompt continuation, cancellation callbacks, or
  /// deadlines. The members are queued atomically in the given order under one
  /// scheduler cohort identity. This is a dormant admission and order contract:
  /// members still execute as independent C1 work, and no distributed runner
  /// plan or physical C2 collective is selected here.
  [[nodiscard]] Cohort SubmitCohort(std::vector<CohortMemberRequest> members);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

using TextRequestMetadata = TextGenerationScheduler::RequestMetadata;
using TextRequestCohort = TextGenerationScheduler::Cohort;
using TextCohortMemberRequest = TextGenerationScheduler::CohortMemberRequest;

}  // namespace gufo::server

#endif  // GUFO_SERVER_TEXT_GENERATION_SCHEDULER_HPP_

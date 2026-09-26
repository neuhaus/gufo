#ifndef GUFO_SERVER_TP_COHORT_WORKER_HPP_
#define GUFO_SERVER_TP_COHORT_WORKER_HPP_

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/cli/serve/tp_control.hpp"

namespace gufo::server {

/// What one worker-loop command does to the surrounding receive loop.
enum class TpWorkerLoopStep {
  /// A wire-valid response was sent; keep receiving commands.
  kContinue,
  /// The control channel or the operation scope is unusable; stop the worker.
  kStop,
};

/// The command-sequence operation scope a C2 cohort runs under.
///
/// `Begin` and `End` are the only scope operations this seam performs, and the
/// sequence guarantees that a successful `Begin` is matched by exactly one
/// `End`, so a leaked scope can never poison the next C1 command's bind.
class TpCohortLease {
public:
  virtual ~TpCohortLease() = default;

  /// Binds the operation scope. A refused bind owns nothing, so `End` must not
  /// follow it.
  [[nodiscard]] virtual bool Begin(std::string* error) = 0;
  /// Releases a bound operation scope.
  [[nodiscard]] virtual bool End(std::string* error) = 0;
};

/// One admitted cohort member as seen by the worker loop.
class TpCohortMemberHandle {
public:
  virtual ~TpCohortMemberHandle() = default;

  /// Waits for the member's terminal result, exactly like a C1 `Request::Wait`.
  [[nodiscard]] virtual TextGenerationBackend::Result Wait() = 0;
  /// Cancels a still-running member.
  virtual void Cancel() = 0;
};

/// One atomically admitted, ordered C2 cohort.
class TpCohortSubmission {
public:
  virtual ~TpCohortSubmission() = default;

  [[nodiscard]] virtual TpCohortMemberHandle& Member(std::size_t index) = 0;
};

/// Worker-loop environment for one dormant C2 command.
struct TpCohortWorkerHooks {
  /// Admits both members atomically. Returns nullptr and sets error on refusal.
  std::function<std::unique_ptr<TpCohortSubmission>(
      std::vector<TextGenerationScheduler::CohortMemberRequest>, std::string*)>
      submit;
  /// Sends one wire response; kStop only when the channel is unusable.
  std::function<TpWorkerLoopStep(const TpControlResponse&, std::string*)> send;
};

/// Runs one dormant `kCohort2Ar` command on the rank-1 worker loop.
///
/// The cohort is admitted as one ordered pair and executed as two serial C1
/// members under exactly one lease scoped to the command sequence. No shared C2
/// collective is bound and no member may report a parallel plan, so this path
/// stays dormant: the coordinator sends no `kCohort2Ar` command, and the
/// response seam rejects any member that is not serial. MTP drafts cannot be
/// represented by the C2 response contract, so a loaded draft sidecar fails
/// closed before the lease is bound.
///
/// Serial execution also depends on `runner_capacity`, the worker scheduler's
/// runner capacity. `Admit` admits up to that many requests at once
/// (text_generation_scheduler.cpp:491), so above one both cohort members become
/// co-resident.
///
/// They would then interleave, but not through the batched path: that requires
/// `multi_token_decode`, which is `use_mtp_` for this runner
/// (inference_backend.cpp:2503) and therefore false for a C2 AR cohort, which
/// refuses MTP outright. Each resident member instead advances through the
/// per-request `StepDecode` (text_generation_scheduler.cpp:709), one at a time,
/// in whatever order the scheduler selects.
///
/// That order is not carried on the wire, so the two ranks' collective
/// sequences stop being guaranteed to match. A mismatch would also be silent: a
/// single-row decode is `hidden * 4` bytes for either member, so ordinal N on
/// one rank can pair with ordinal N on the other for a *different* member
/// without tripping the communicator's scope/ordinal/bytes header check. Any
/// capacity other than one therefore fails closed before the lease is bound.
[[nodiscard]] TpWorkerLoopStep RunTpCohortCommand(
    const TpControlCommand& command, bool use_mtp, std::size_t runner_capacity,
    std::unique_ptr<TpCohortLease> lease, TpCohortWorkerHooks hooks,
    std::string* error);

}  // namespace gufo::server

#endif  // GUFO_SERVER_TP_COHORT_WORKER_HPP_

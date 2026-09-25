#include "src/cli/serve/tp_cohort_worker.hpp"

#include <array>
#include <string_view>
#include <utility>

#include "src/cli/serve/tp_cohort_plan.hpp"

namespace gufo::server {
namespace {

/// Mirrors the worker protocol error byte bound in tp_control.cpp.
constexpr std::size_t kMaxWorkerErrorBytes = 1U << 20;
constexpr std::string_view kBindFailure =
    "TP worker operation scope bind failed: ";
constexpr std::string_view kCleanupFailure =
    "TP worker operation scope cleanup failed: ";
/// MTP draft telemetry has no representation in the C2 response contract.
constexpr std::string_view kMtpRefusal =
    "C2 cohort AR is not supported while the MTP sidecar is loaded";
/// Serial members need a worker scheduler that runs one request at a time.
constexpr std::string_view kRunnerCapacityRefusal =
    "C2 cohort AR requires worker runner capacity 1, got ";

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

/// The single release site for one C2 operation scope.
///
/// Every exit path after a successful `Begin` runs through `End`, and `End` is
/// the only place that reaches the lease, so a bound scope is released exactly
/// once no matter how the cohort exits. A refused `Begin` owns nothing, so it
/// leaves the guard unbound and is never released. The destructor is a
/// no-poison safety net for an exception that unwinds past a return, never a
/// second release.
class TpCohortLeaseGuard {
public:
  explicit TpCohortLeaseGuard(std::unique_ptr<TpCohortLease> lease)
      : lease_(std::move(lease)) {}
  ~TpCohortLeaseGuard() { (void)End(nullptr); }

  TpCohortLeaseGuard(const TpCohortLeaseGuard&) = delete;
  TpCohortLeaseGuard& operator=(const TpCohortLeaseGuard&) = delete;
  TpCohortLeaseGuard(TpCohortLeaseGuard&&) = delete;
  TpCohortLeaseGuard& operator=(TpCohortLeaseGuard&&) = delete;

  [[nodiscard]] bool Begin(std::string* error) {
    if (lease_ == nullptr) {
      SetError(error, "TP worker operation scope is missing");
      return false;
    }
    if (!lease_->Begin(error)) {
      return false;
    }
    bound_ = true;
    return true;
  }

  /// Releases a bound scope at most once; a later call is a no-op.
  [[nodiscard]] bool End(std::string* error) {
    if (!bound_ || lease_ == nullptr) {
      return true;
    }
    bound_ = false;
    return lease_->End(error);
  }

private:
  std::unique_ptr<TpCohortLease> lease_;
  bool bound_{false};
};

/// Copies the command's own cohort identity into a plan used for failure
/// responses.
///
/// `ReceiveCommand` already accepted the C2 envelope, so the sequence, cohort
/// ID, plan digests and the two ordered member IDs are wire-valid here. The
/// copy still bounds the member index because it runs on the rejection path,
/// where the plan seam refused the command.
[[nodiscard]] TpCohortPlan CohortIdentityFromCommand(
    const TpControlCommand& command) {
  TpCohortPlan plan{
      .sequence = command.sequence,
      .cohort_id = command.cohort_id,
      .execution_plan_digest = command.execution_plan_digest,
      .cache_plan_digest = command.cache_plan_digest,
  };
  for (std::size_t index = 0; index < kCohort2MemberCount; ++index) {
    if (index < command.members.size()) {
      plan.member_ids[index] = command.members[index].member_id;
    }
  }
  return plan;
}

/// Appends one scope-cleanup poison to a response error, bounded by the worker
/// protocol error byte limit.
[[nodiscard]] std::string AppendCleanupFailure(std::string base,
                                               std::string_view reason) {
  if (!base.empty()) {
    base += "; ";
  }
  base += kCleanupFailure;
  base += reason;
  if (base.size() > kMaxWorkerErrorBytes) {
    base.resize(kMaxWorkerErrorBytes);
  }
  return base;
}

/// Sends one C2 rejection that left no operation scope behind. A rejected
/// cohort is a translation or local-precondition error, not a worker fault, so
/// the worker keeps serving; only a lost control channel stops it.
[[nodiscard]] TpWorkerLoopStep SendCohortFailure(
    const TpCohortPlan& plan, std::string_view reason,
    const TpCohortWorkerHooks& hooks, std::string* error) {
  return hooks.send(BuildCohortFailureResponse(plan, reason), error);
}

/// Releases the bound scope, then sends one C2 response.
///
/// C1 ends its scope before it answers, so the cohort mirrors that order: no
/// runner is left decoding behind a response that is already on the wire. A
/// cleanup failure still sends the response, poisons its error with the reason
/// the worker stopped, and stops the worker exactly as the C1 tail does; a lost
/// control channel stops it too.
[[nodiscard]] TpWorkerLoopStep FinalizeCohort(TpControlResponse response,
                                              TpCohortLeaseGuard& operation,
                                              const TpCohortWorkerHooks& hooks,
                                              std::string* error) {
  std::string operation_error;
  const bool released = operation.End(&operation_error);
  if (!released) {
    response.error =
        AppendCleanupFailure(std::move(response.error), operation_error);
  }
  if (hooks.send(response, error) == TpWorkerLoopStep::kStop) {
    return TpWorkerLoopStep::kStop;
  }
  if (!released) {
    SetError(error, std::move(response.error));
    return TpWorkerLoopStep::kStop;
  }
  return TpWorkerLoopStep::kContinue;
}

}  // namespace

TpWorkerLoopStep RunTpCohortCommand(const TpControlCommand& command,
                                    bool use_mtp, std::size_t runner_capacity,
                                    std::unique_ptr<TpCohortLease> lease,
                                    TpCohortWorkerHooks hooks,
                                    std::string* error) {
  TpCohortLeaseGuard operation(std::move(lease));
  std::string cohort_error;
  auto plan = BuildCohortPlan(command, &cohort_error);
  if (!plan.has_value()) {
    return SendCohortFailure(CohortIdentityFromCommand(command), cohort_error,
                             hooks, error);
  }
  if (use_mtp) {
    return SendCohortFailure(*plan, kMtpRefusal, hooks, error);
  }
  // A wider runner pool would make both members co-resident, and the
  // scheduler would interleave them in an order the wire does not carry, so the
  // ranks' collective sequences could pair different members at equal byte
  // counts without a header error. Refuse before binding and before any
  // collective. See the header for why the batched path is not the mechanism.
  if (runner_capacity != 1) {
    const std::string refusal =
        std::string(kRunnerCapacityRefusal) + std::to_string(runner_capacity);
    return SendCohortFailure(*plan, refusal, hooks, error);
  }
  auto members = BuildCohortMembers(command, &cohort_error);
  if (!members.has_value()) {
    return SendCohortFailure(*plan, cohort_error, hooks, error);
  }
  std::vector<TextGenerationScheduler::CohortMemberRequest> admissions;
  admissions.reserve(members->size());
  for (auto& member : *members) {
    admissions.push_back(std::move(member));
  }
  // The scope is bound before admission, exactly like C1, so a queued member
  // can never reach a runner with an unbound scope. One lease spans the whole
  // cohort: the C2 members share the command sequence scope, so a per-member
  // lease would open and close the same collective twice.
  std::string operation_error;
  if (!operation.Begin(&operation_error)) {
    const auto response = BuildCohortFailureResponse(
        *plan, std::string(kBindFailure) + operation_error);
    if (hooks.send(response, error) == TpWorkerLoopStep::kStop) {
      return TpWorkerLoopStep::kStop;
    }
    // The scope could not be bound, so rank 0 can never be answered: the worker
    // stops exactly as the C1 path does, and a refused bind is never released.
    SetError(error, response.error);
    return TpWorkerLoopStep::kStop;
  }

  // From here on the scope is bound, so every exit below releases it through
  // `operation` before the response leaves the worker.
  std::string admission_error;
  std::unique_ptr<TpCohortSubmission> submission =
      hooks.submit(std::move(admissions), &admission_error);
  if (submission == nullptr) {
    // Admission refused the cohort before either member reached a runner, so
    // the scope is released before the worker keeps serving. A leaked scope
    // would poison the next command's Begin(). A failed release leaves the
    // worker in an unknown scope state, so it stops without answering rank 0.
    if (!operation.End(&operation_error)) {
      SetError(error, std::string(kCleanupFailure) + operation_error);
      return TpWorkerLoopStep::kStop;
    }
    return SendCohortFailure(*plan, admission_error, hooks, error);
  }

  std::array<TextGenerationBackend::Result, kCohort2MemberCount> results;
  std::size_t member_index = 0;
  try {
    for (; member_index < kCohort2MemberCount; ++member_index) {
      results[member_index] = submission->Member(member_index).Wait();
    }
  } catch (const std::exception& exception) {
    // The member that threw is finished; cancel its peer so a rejected cohort
    // never leaves a runner decoding behind a sent C2 error.
    if (member_index + 1 < kCohort2MemberCount) {
      submission->Member(member_index + 1).Cancel();
    }
    return FinalizeCohort(BuildCohortFailureResponse(*plan, exception.what()),
                          operation, hooks, error);
  }
  // Both members are terminal, so the admission handles are released before the
  // response is built, exactly as the dormant C2 worker released its cohort.
  submission.reset();
  const auto translation = BuildCohortResponse(*plan, results, &cohort_error);
  return FinalizeCohort(translation.response.has_value() ? *translation.response
                                                         : translation.failure,
                        operation, hooks, error);
}

}  // namespace gufo::server

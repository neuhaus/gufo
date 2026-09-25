#include "src/cli/serve/tp_cohort_worker.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/cli/serve/tp_cohort_plan.hpp"
#include "src/cli/serve/tp_control.hpp"

namespace {

using gufo::server::ComputeTpCachePlanDigest;
using gufo::server::ComputeTpExecutionPlanDigest;
using gufo::server::kCohort2MemberCount;
using gufo::server::RunTpCohortCommand;
using gufo::server::TextGenerationBackend;
using gufo::server::TextGenerationScheduler;
using gufo::server::TpCohortLease;
using gufo::server::TpCohortMemberHandle;
using gufo::server::TpCohortSubmission;
using gufo::server::TpCohortWorkerHooks;
using gufo::server::TpControlCommand;
using gufo::server::TpControlCommandKind;
using gufo::server::TpControlResponse;
using gufo::server::TpControlResponseKind;
using gufo::server::TpWorkerLoopStep;
using gufo::server::ValidateTpControlCommand;
using gufo::server::ValidateTpControlResponse;

using MemberResults =
    std::array<TextGenerationBackend::Result, kCohort2MemberCount>;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    std::exit(1);
  }
}

/// One worker-loop event, recorded in the order the seam performed it.
using EventLog = std::vector<std::string>;

/// Every counter a test needs, shared with the seam's fakes. The fakes are
/// owned by the seam and die with the command, so their state lives here
/// instead: a leak, a double release or a stray cancel stays visible after the
/// run.
struct Recorder {
  EventLog log;
  int begin_calls{0};
  int end_calls{0};
  std::array<int, kCohort2MemberCount> cancel_calls{};
};

std::string Join(const EventLog& log) {
  std::string joined;
  for (const auto& event : log) {
    if (!joined.empty()) {
      joined += ",";
    }
    joined += event;
  }
  return joined;
}

bool Matches(const EventLog& log, const EventLog& expected) {
  return log == expected;
}

/// Scriptable operation scope that counts every bind and release it is asked
/// for, so a leak or a double release is visible from the test side. It has no
/// destructor-side release of its own: only the seam can end a scope.
class FakeLease final : public TpCohortLease {
public:
  FakeLease(std::shared_ptr<Recorder> recorder, bool begin_ok, bool end_ok)
      : recorder_(std::move(recorder)), begin_ok_(begin_ok), end_ok_(end_ok) {}

  [[nodiscard]] bool Begin(std::string* error) override {
    recorder_->log.push_back("begin");
    ++recorder_->begin_calls;
    if (!begin_ok_) {
      if (error != nullptr) {
        *error = "scope already active";
      }
      return false;
    }
    return true;
  }

  [[nodiscard]] bool End(std::string* error) override {
    recorder_->log.push_back("end");
    ++recorder_->end_calls;
    if (!end_ok_) {
      if (error != nullptr) {
        *error = "scope cleanup refused";
      }
      return false;
    }
    return true;
  }

private:
  std::shared_ptr<Recorder> recorder_;
  bool begin_ok_{true};
  bool end_ok_{true};
};

/// Scriptable cohort member. `Wait` can throw and `Cancel` is recorded, so a
/// rejected cohort can be proven to cancel exactly the peer it must.
class FakeMember final : public TpCohortMemberHandle {
public:
  FakeMember(std::shared_ptr<Recorder> recorder, std::size_t index, bool throws,
             TextGenerationBackend::Result result)
      : recorder_(std::move(recorder)),
        index_(index),
        throws_(throws),
        result_(std::move(result)) {}

  [[nodiscard]] TextGenerationBackend::Result Wait() override {
    recorder_->log.push_back("wait:" + std::to_string(index_));
    if (throws_) {
      throw std::runtime_error("cohort member wait failed");
    }
    return result_;
  }

  void Cancel() override {
    recorder_->log.push_back("cancel:" + std::to_string(index_));
    ++recorder_->cancel_calls[index_];
  }

private:
  std::shared_ptr<Recorder> recorder_;
  std::size_t index_{0};
  bool throws_{false};
  TextGenerationBackend::Result result_;
};

/// One admitted cohort of two scriptable members.
class FakeSubmission final : public TpCohortSubmission {
public:
  FakeSubmission(std::shared_ptr<Recorder> recorder,
                 std::array<bool, kCohort2MemberCount> throws,
                 const MemberResults& results)
      : recorder_(std::move(recorder)) {
    for (std::size_t index = 0; index < kCohort2MemberCount; ++index) {
      members_[index] = std::make_unique<FakeMember>(
          recorder_, index, throws[index], results[index]);
    }
  }

  [[nodiscard]] TpCohortMemberHandle& Member(std::size_t index) override {
    return *members_[index];
  }

private:
  std::shared_ptr<Recorder> recorder_;
  std::array<std::unique_ptr<FakeMember>, kCohort2MemberCount> members_;
};

/// Scriptable worker environment: every knob the worker loop can turn.
struct Options {
  bool use_mtp{false};
  bool begin_ok{true};
  bool end_ok{true};
  bool submit_ok{true};
  std::string submit_error{"cohort admission refused"};
  std::array<bool, kCohort2MemberCount> member_throws{{false, false}};
  MemberResults results;
  TpWorkerLoopStep send_step{TpWorkerLoopStep::kContinue};
};

/// What one worker-loop run produced.
struct Outcome {
  TpWorkerLoopStep step{TpWorkerLoopStep::kStop};
  std::string error;
  std::vector<TpControlResponse> responses;
  std::vector<std::vector<std::uint64_t>> admitted_member_ids;
  std::shared_ptr<Recorder> recorder;
};

TextGenerationBackend::Result MakeSerialResult(
    std::vector<gufo::tokenization::TokenId> tokens,
    std::string execution_plan) {
  TextGenerationBackend::Result result;
  result.tokens = std::move(tokens);
  result.execution_plan = std::move(execution_plan);
  result.physical_execution_width = 1;
  result.completion_tokens = result.tokens.size();
  return result;
}

MemberResults MakeSerialResults() {
  return {MakeSerialResult({100, 101}, "serial-c1"),
          MakeSerialResult({200, 201, 202}, "serial-fallback")};
}

void SealCommand(TpControlCommand* command) {
  command->execution_plan_digest = ComputeTpExecutionPlanDigest(*command);
  command->cache_plan_digest = ComputeTpCachePlanDigest(*command);
}

TpControlCommand MakeC2Command(std::uint64_t sequence,
                               std::uint64_t cohort_id) {
  TpControlCommand command{
      .sequence = sequence,
      .kind = TpControlCommandKind::kCohort2Ar,
      .cohort_id = cohort_id,
      .members =
          {
              {
                  .member_id = cohort_id + 1,
                  .max_tokens = 4,
                  .prompt_tokens = {10, 11, 12},
                  .client_id = "member-zero",
              },
              {
                  .member_id = cohort_id + 2,
                  .max_tokens = 6,
                  .prompt_tokens = {20, 21},
                  .client_id = "member-one",
              },
          },
  };
  SealCommand(&command);
  return command;
}

Options MakeOptions() {
  Options options;
  options.results = MakeSerialResults();
  return options;
}

/// Runs one command through the seam with a fully scripted environment. The
/// fake lease and submission stay shared, so their counters outlive the run and
/// the test sees every bind, release and cancel the seam asked for.
Outcome RunCommand(const TpControlCommand& command, const Options& options) {
  Outcome outcome;
  outcome.recorder = std::make_shared<Recorder>();
  auto recorder = outcome.recorder;

  TpCohortWorkerHooks hooks;
  using Admissions = std::vector<TextGenerationScheduler::CohortMemberRequest>;
  hooks.submit =
      [recorder, options, &outcome](
          Admissions admissions,
          std::string* error) -> std::unique_ptr<TpCohortSubmission> {
    std::vector<std::uint64_t> member_ids;
    member_ids.reserve(admissions.size());
    for (const auto& admission : admissions) {
      member_ids.push_back(admission.member_id);
    }
    recorder->log.push_back("submit:" + std::to_string(admissions.size()));
    outcome.admitted_member_ids.push_back(std::move(member_ids));
    if (!options.submit_ok) {
      if (error != nullptr) {
        *error = options.submit_error;
      }
      return nullptr;
    }
    return std::make_unique<FakeSubmission>(recorder, options.member_throws,
                                            options.results);
  };
  hooks.send = [recorder, options, &outcome](const TpControlResponse& response,
                                             std::string* error) {
    recorder->log.push_back("send");
    outcome.responses.push_back(response);
    if (options.send_step == TpWorkerLoopStep::kStop && error != nullptr) {
      *error = "control channel closed";
    }
    return options.send_step;
  };

  std::string error;
  outcome.step = RunTpCohortCommand(
      command, options.use_mtp,
      std::make_unique<FakeLease>(recorder, options.begin_ok, options.end_ok),
      std::move(hooks), &error);
  outcome.error = std::move(error);
  return outcome;
}

/// The lease accounting invariant of the seam: a refused bind is never
/// released, and a bound scope is released exactly once on every exit path.
void RequireLease(const Outcome& outcome, int begins, int ends,
                  const std::string& what) {
  Require(outcome.recorder->begin_calls == begins,
          what + ": binds the operation scope " + std::to_string(begins) +
              " time(s), saw " + std::to_string(outcome.recorder->begin_calls));
  Require(outcome.recorder->end_calls == ends,
          what + ": releases the operation scope " + std::to_string(ends) +
              " time(s), saw " + std::to_string(outcome.recorder->end_calls));
}

void RequireWireValid(const TpControlResponse& response,
                      const std::string& what) {
  std::string error;
  Require(ValidateTpControlResponse(response, &error),
          what + " is a wire-valid C2 response: " + error);
}

void RequireSingleFailure(const Outcome& outcome, const std::string& what) {
  Require(outcome.responses.size() == 1,
          what + ": sends exactly one response, saw " +
              std::to_string(outcome.responses.size()));
  Require(!outcome.responses.front().error.empty(),
          what + ": reports a non-empty failure reason");
  Require(outcome.responses.front().kind == TpControlResponseKind::kCohort2Ar,
          what + ": keeps the C2 response envelope");
  Require(outcome.responses.front().members.size() == kCohort2MemberCount,
          what + ": never truncates the cohort to a partial envelope");
}

void TestSuccessOrdersMembersAndReleasesOnce() {
  const auto command = MakeC2Command(11, 300);
  const Outcome outcome = RunCommand(command, MakeOptions());
  RequireLease(outcome, 1, 1, "TP cohort worker success");
  Require(Matches(outcome.recorder->log,
                  {"begin", "submit:2", "wait:0", "wait:1", "end", "send"}),
          "TP cohort worker waits both members in order, then releases the "
          "scope, then answers: " +
              Join(outcome.recorder->log));
  Require(outcome.step == TpWorkerLoopStep::kContinue && outcome.error.empty(),
          "TP cohort worker keeps serving after a successful cohort");
  Require(outcome.admitted_member_ids.size() == 1 &&
              outcome.admitted_member_ids.front() ==
                  std::vector<std::uint64_t>{command.members[0].member_id,
                                             command.members[1].member_id},
          "TP cohort worker admits both members in command order");
  Require(outcome.responses.size() == 1,
          "TP cohort worker answers a successful cohort exactly once");
  const auto& response = outcome.responses.front();
  Require(response.sequence == command.sequence &&
              response.cohort_id == command.cohort_id &&
              response.execution_plan_digest == command.execution_plan_digest &&
              response.cache_plan_digest == command.cache_plan_digest &&
              response.error.empty(),
          "TP cohort worker echoes the accepted cohort envelope");
  Require(
      response.members.size() == kCohort2MemberCount &&
          response.members[0].member_id == command.members[0].member_id &&
          response.members[1].member_id == command.members[1].member_id &&
          response.members[0].tokens == std::vector<std::int32_t>{100, 101} &&
          response.members[1].tokens ==
              std::vector<std::int32_t>{200, 201, 202},
      "TP cohort worker maps member results onto the ordered identities");
  RequireWireValid(response, "TP cohort worker success response");
}

void TestMtpRefusalFailsClosedBeforeBinding() {
  auto options = MakeOptions();
  options.use_mtp = true;
  const auto command = MakeC2Command(12, 310);
  const Outcome outcome = RunCommand(command, options);
  RequireLease(outcome, 0, 0, "TP cohort worker with a draft sidecar");
  Require(Matches(outcome.recorder->log, {"send"}),
          "TP cohort worker fails closed before it admits or binds: " +
              Join(outcome.recorder->log));
  Require(outcome.step == TpWorkerLoopStep::kContinue && outcome.error.empty(),
          "TP cohort worker keeps serving after an MTP refusal");
  RequireSingleFailure(outcome, "TP cohort worker MTP refusal");
  Require(outcome.responses.front().error ==
              "C2 cohort AR is not supported while the MTP sidecar is loaded",
          "TP cohort worker refuses C2 while the MTP sidecar is loaded");
  Require(outcome.responses.front().members[0].member_id ==
                  command.members[0].member_id &&
              outcome.responses.front().members[1].member_id ==
                  command.members[1].member_id,
          "TP cohort worker MTP refusal keeps both ordered member identities");
  RequireWireValid(outcome.responses.front(),
                   "TP cohort worker MTP refusal response");
}

void TestMemberTranslationRejectionNeverBindsTheScope() {
  // A member token budget the members seam refuses. The members seam
  // re-validates the envelope it is handed, so the worker rejects before it
  // binds either way; what this pins is the observable contract of the pre-bind
  // rejection: no scope is opened, one C2 rejection is reported, and the worker
  // keeps serving.
  auto command = MakeC2Command(13, 320);
  command.members[1].max_tokens = 0;
  SealCommand(&command);
  const Outcome outcome = RunCommand(command, MakeOptions());
  RequireLease(outcome, 0, 0, "TP cohort worker with a refused member");
  Require(Matches(outcome.recorder->log, {"send"}),
          "TP cohort worker rejects a refused member before it binds: " +
              Join(outcome.recorder->log));
  Require(outcome.step == TpWorkerLoopStep::kContinue && outcome.error.empty(),
          "TP cohort worker keeps serving after a member translation refusal");
  RequireSingleFailure(outcome, "TP cohort worker member translation refusal");
  Require(outcome.responses.front().members[0].member_id ==
                  command.members[0].member_id &&
              outcome.responses.front().members[1].member_id ==
                  command.members[1].member_id,
          "TP cohort worker member refusal keeps both ordered identities");
}

void TestPlanRejectionReportsCommandIdentity() {
  auto command = MakeC2Command(14, 330);
  command.execution_plan_digest = {};
  const Outcome outcome = RunCommand(command, MakeOptions());
  RequireLease(outcome, 0, 0, "TP cohort worker with a rejected command");
  Require(Matches(outcome.recorder->log, {"send"}),
          "TP cohort worker rejects an invalid command before it binds: " +
              Join(outcome.recorder->log));
  Require(outcome.step == TpWorkerLoopStep::kContinue && outcome.error.empty(),
          "TP cohort worker keeps serving after a command rejection");
  RequireSingleFailure(outcome, "TP cohort worker command rejection");
  const auto& response = outcome.responses.front();
  // A refused plan reports the command's own identity, so rank 0 can still
  // correlate the rejection. The refused plan digests are echoed verbatim,
  // which keeps this envelope outside the accepted-response contract.
  Require(response.sequence == command.sequence &&
              response.cohort_id == command.cohort_id &&
              response.cache_plan_digest == command.cache_plan_digest &&
              response.members[0].member_id == command.members[0].member_id &&
              response.members[1].member_id == command.members[1].member_id,
          "TP cohort worker reports a rejected command under its own identity");
}

void TestBindFailureStopsWithoutReleasing() {
  auto options = MakeOptions();
  options.begin_ok = false;
  const auto command = MakeC2Command(15, 340);
  const Outcome outcome = RunCommand(command, options);
  RequireLease(outcome, 1, 0, "TP cohort worker with a refused bind");
  Require(Matches(outcome.recorder->log, {"begin", "send"}),
          "TP cohort worker stops on a refused bind without releasing: " +
              Join(outcome.recorder->log));
  Require(outcome.step == TpWorkerLoopStep::kStop,
          "TP cohort worker stops the worker loop on a refused bind");
  RequireSingleFailure(outcome, "TP cohort worker refused bind");
  Require(outcome.responses.front().error ==
              "TP worker operation scope bind failed: scope already active",
          "TP cohort worker reports why the scope could not be bound");
  Require(outcome.error == outcome.responses.front().error,
          "TP cohort worker poisons its local error with the bind failure: " +
              outcome.error);
  Require(outcome.error.size() <= (1U << 20),
          "TP cohort worker keeps the bind failure inside the error byte "
          "limit");
  RequireWireValid(outcome.responses.front(),
                   "TP cohort worker refused bind response");
}

void TestSubmitRefusalReleasesTheScopeAndContinues() {
  auto options = MakeOptions();
  options.submit_ok = false;
  const auto command = MakeC2Command(16, 350);
  const Outcome outcome = RunCommand(command, options);
  RequireLease(outcome, 1, 1, "TP cohort worker with a refused admission");
  Require(Matches(outcome.recorder->log, {"begin", "submit:2", "end", "send"}),
          "TP cohort worker releases a refused admission before answering: " +
              Join(outcome.recorder->log));
  Require(outcome.step == TpWorkerLoopStep::kContinue && outcome.error.empty(),
          "TP cohort worker keeps serving after a refused admission");
  RequireSingleFailure(outcome, "TP cohort worker refused admission");
  Require(outcome.responses.front().error == options.submit_error,
          "TP cohort worker reports the admission refusal: " +
              outcome.responses.front().error);
  Require(outcome.responses.front().members[0].member_id ==
                  command.members[0].member_id &&
              outcome.responses.front().members[1].member_id ==
                  command.members[1].member_id,
          "TP cohort worker admission refusal keeps both ordered identities");
  RequireWireValid(outcome.responses.front(),
                   "TP cohort worker refused admission response");
}

void TestSubmitRefusalWithReleaseFailureStopsWithoutAnswering() {
  auto options = MakeOptions();
  options.submit_ok = false;
  options.end_ok = false;
  const Outcome outcome = RunCommand(MakeC2Command(17, 360), options);
  RequireLease(outcome, 1, 1, "TP cohort worker with a failed release");
  Require(Matches(outcome.recorder->log, {"begin", "submit:2", "end"}),
          "TP cohort worker stops on a failed release without answering: " +
              Join(outcome.recorder->log));
  Require(outcome.responses.empty(),
          "TP cohort worker never answers rank 0 from an unknown scope state");
  Require(outcome.step == TpWorkerLoopStep::kStop &&
              outcome.error ==
                  "TP worker operation scope cleanup failed: scope cleanup "
                  "refused",
          "TP cohort worker poisons its local error with the release "
          "failure: " +
              outcome.error);
}

void TestFirstMemberThrowCancelsThePeerAndReleasesOnce() {
  auto options = MakeOptions();
  options.member_throws = {true, false};
  const auto command = MakeC2Command(18, 370);
  const Outcome outcome = RunCommand(command, options);
  RequireLease(outcome, 1, 1, "TP cohort worker with a failed first member");
  Require(Matches(outcome.recorder->log,
                  {"begin", "submit:2", "wait:0", "cancel:1", "end", "send"}),
          "TP cohort worker cancels the peer, releases the scope, then "
          "answers: " +
              Join(outcome.recorder->log));
  Require(outcome.recorder->cancel_calls[0] == 0 &&
              outcome.recorder->cancel_calls[1] == 1,
          "TP cohort worker cancels only the unfinished peer");
  Require(outcome.step == TpWorkerLoopStep::kContinue && outcome.error.empty(),
          "TP cohort worker keeps serving after a member wait failure");
  RequireSingleFailure(outcome, "TP cohort worker member wait failure");
  Require(outcome.responses.front().error == "cohort member wait failed",
          "TP cohort worker reports the member failure: " +
              outcome.responses.front().error);
  RequireWireValid(outcome.responses.front(),
                   "TP cohort worker member wait failure response");
}

void TestSecondMemberThrowCancelsNothingAndReleasesOnce() {
  auto options = MakeOptions();
  options.member_throws = {false, true};
  const auto command = MakeC2Command(19, 380);
  const Outcome outcome = RunCommand(command, options);
  RequireLease(outcome, 1, 1, "TP cohort worker with a failed second member");
  Require(Matches(outcome.recorder->log,
                  {"begin", "submit:2", "wait:0", "wait:1", "end", "send"}),
          "TP cohort worker cancels no member once the last one failed: " +
              Join(outcome.recorder->log));
  Require(outcome.recorder->cancel_calls[0] == 0 &&
              outcome.recorder->cancel_calls[1] == 0,
          "TP cohort worker cancels nothing when the last member failed");
  Require(outcome.step == TpWorkerLoopStep::kContinue && outcome.error.empty(),
          "TP cohort worker keeps serving after the last member failed");
  RequireSingleFailure(outcome, "TP cohort worker last member failure");
  Require(outcome.responses.front().members[0].member_id ==
                  command.members[0].member_id &&
              outcome.responses.front().members[1].member_id ==
                  command.members[1].member_id,
          "TP cohort worker member failure keeps both ordered identities");
}

void TestRejectedMemberResultStillReleasesOnce() {
  auto options = MakeOptions();
  options.results[0].physical_execution_width = 2;
  const Outcome outcome = RunCommand(MakeC2Command(20, 390), options);
  RequireLease(outcome, 1, 1, "TP cohort worker with a widened member");
  Require(Matches(outcome.recorder->log,
                  {"begin", "submit:2", "wait:0", "wait:1", "end", "send"}),
          "TP cohort worker releases a rejected member result once: " +
              Join(outcome.recorder->log));
  Require(outcome.step == TpWorkerLoopStep::kContinue && outcome.error.empty(),
          "TP cohort worker keeps serving after a rejected member result");
  RequireSingleFailure(outcome, "TP cohort worker rejected member result");
  Require(outcome.responses.front().error ==
              "TP cohort members must report serial width and a serial "
              "execution plan",
          "TP cohort worker reports why the member result was rejected: " +
              outcome.responses.front().error);
  RequireWireValid(outcome.responses.front(),
                   "TP cohort worker rejected member result response");
}

void TestReleaseFailureOnSuccessStillAnswersAndStops() {
  auto options = MakeOptions();
  options.end_ok = false;
  const Outcome outcome = RunCommand(MakeC2Command(21, 400), options);
  RequireLease(outcome, 1, 1, "TP cohort worker with a failed release");
  Require(Matches(outcome.recorder->log,
                  {"begin", "submit:2", "wait:0", "wait:1", "end", "send"}),
          "TP cohort worker answers a cohort whose release failed: " +
              Join(outcome.recorder->log));
  Require(outcome.responses.size() == 1,
          "TP cohort worker still sends the cohort response it already built");
  const auto& response = outcome.responses.front();
  Require(
      response.members.size() == kCohort2MemberCount &&
          response.members[0].tokens == std::vector<std::int32_t>{100, 101} &&
          response.members[1].tokens ==
              std::vector<std::int32_t>{200, 201, 202},
      "TP cohort worker keeps the member results on a poisoned response");
  Require(response.error ==
              "TP worker operation scope cleanup failed: scope cleanup refused",
          "TP cohort worker poisons the response with the release failure: " +
              response.error);
  Require(
      outcome.step == TpWorkerLoopStep::kStop &&
          outcome.error == response.error,
      "TP cohort worker stops with the same poison it sent: " + outcome.error);
  RequireWireValid(response, "TP cohort worker poisoned success response");
}

void TestSendFailureStillReleasesOnce() {
  auto options = MakeOptions();
  options.send_step = TpWorkerLoopStep::kStop;
  const Outcome outcome = RunCommand(MakeC2Command(22, 410), options);
  RequireLease(outcome, 1, 1, "TP cohort worker with a lost control channel");
  Require(Matches(outcome.recorder->log,
                  {"begin", "submit:2", "wait:0", "wait:1", "end", "send"}),
          "TP cohort worker releases the scope even when the channel dies: " +
              Join(outcome.recorder->log));
  Require(outcome.responses.size() == 1,
          "TP cohort worker attempts the response exactly once");
  Require(outcome.step == TpWorkerLoopStep::kStop &&
              outcome.error == "control channel closed",
          "TP cohort worker stops on a lost control channel: " + outcome.error);
}

}  // namespace

int main() {
  const auto command = MakeC2Command(1, 100);
  std::string error;
  Require(ValidateTpControlCommand(command, &error),
          "TP cohort worker test fixture is a valid C2 command: " + error);

  TestSuccessOrdersMembersAndReleasesOnce();
  TestMtpRefusalFailsClosedBeforeBinding();
  TestMemberTranslationRejectionNeverBindsTheScope();
  TestPlanRejectionReportsCommandIdentity();
  TestBindFailureStopsWithoutReleasing();
  TestSubmitRefusalReleasesTheScopeAndContinues();
  TestSubmitRefusalWithReleaseFailureStopsWithoutAnswering();
  TestFirstMemberThrowCancelsThePeerAndReleasesOnce();
  TestSecondMemberThrowCancelsNothingAndReleasesOnce();
  TestRejectedMemberResultStillReleasesOnce();
  TestReleaseFailureOnSuccessStillAnswersAndStops();
  TestSendFailureStillReleasesOnce();
  return 0;
}

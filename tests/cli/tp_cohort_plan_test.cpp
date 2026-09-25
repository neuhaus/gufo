#include "src/cli/serve/tp_cohort_plan.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

using gufo::server::BuildCohortFailureResponse;
using gufo::server::BuildCohortMembers;
using gufo::server::BuildCohortPlan;
using gufo::server::BuildCohortResponse;
using gufo::server::ComputeTpCachePlanDigest;
using gufo::server::ComputeTpExecutionPlanDigest;
using gufo::server::kCohort2MemberCount;
using gufo::server::TextCohortMemberRequest;
using gufo::server::TextGenerationBackend;
using gufo::server::TextRunnerToken;
using gufo::server::TpCohortPlan;
using gufo::server::TpControlCommand;
using gufo::server::TpControlCommandKind;
using gufo::server::TpControlResponse;
using gufo::server::TpControlResponseKind;
using gufo::server::ValidateTpControlCommand;
using gufo::server::ValidateTpControlResponse;

using CohortMembers = std::array<TextCohortMemberRequest, kCohort2MemberCount>;
using CohortResults =
    std::array<TextGenerationBackend::Result, kCohort2MemberCount>;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    std::exit(1);
  }
}

/// Recomputes the canonical plan digests so a mutated command reaches the
/// translation seam instead of failing the shared envelope check first.
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

CohortResults MakeSerialResults() {
  return {MakeSerialResult({100, 101}, "serial-c1"),
          MakeSerialResult({200, 201, 202}, "serial-fallback")};
}

void RequireStrictEnvelope(const TpControlResponse& response,
                           const TpCohortPlan& plan, const char* what) {
  const std::string prefix = std::string("TP cohort ") + what;
  Require(response.kind == TpControlResponseKind::kCohort2Ar &&
              response.sequence == plan.sequence &&
              response.cohort_id == plan.cohort_id &&
              response.execution_plan_digest == plan.execution_plan_digest &&
              response.cache_plan_digest == plan.cache_plan_digest &&
              response.members.size() == kCohort2MemberCount,
          prefix + " keeps the cohort envelope");
  Require(response.tokens.empty() && response.draft_tokens == 0 &&
              response.draft_accepted_tokens == 0 &&
              response.cached_prompt_tokens == 0 &&
              response.cache_snapshot_bytes == 0,
          prefix + " leaves legacy, draft and cache fields zero");
  std::string validation_error;
  Require(ValidateTpControlResponse(response, &validation_error),
          prefix + " is a valid worker response: " + validation_error);
}

void RequireFailureTranslation(
    const gufo::server::TpCohortTranslation& translation,
    const TpCohortPlan& plan, const std::string& message,
    bool wire_valid = true) {
  Require(!translation.response.has_value() && !translation.error.empty(),
          message + ": reported failure");
  Require(translation.failure.error == translation.error,
          message + ": failure response carries the reported reason");
  Require(translation.failure.members[0].member_id == plan.member_ids[0] &&
              translation.failure.members[1].member_id == plan.member_ids[1] &&
              translation.failure.members[0].tokens.empty() &&
              translation.failure.members[1].tokens.empty(),
          message + ": failure response keeps both ordered members");
  if (wire_valid) {
    RequireStrictEnvelope(translation.failure, plan, "failure response");
  }
}

void RequireRejectedMembers(const TpControlCommand& command,
                            const std::string& message) {
  std::string error;
  const auto members = BuildCohortMembers(command, &error);
  Require(!members.has_value() && !error.empty(), message);
  const auto plan = BuildCohortPlan(command, &error);
  Require(!plan.has_value(), message + ": rejected before admission");
}

void TestCohortPlanAcceptsValidatedCommand() {
  const auto command = MakeC2Command(41, 900);
  std::string error;
  const auto plan = BuildCohortPlan(command, &error);
  Require(plan.has_value() && error.empty(),
          "TP cohort plan accepts a validated C2 command: " + error);
  Require(plan->sequence == command.sequence &&
              plan->cohort_id == command.cohort_id &&
              plan->execution_plan_digest == command.execution_plan_digest &&
              plan->cache_plan_digest == command.cache_plan_digest &&
              plan->member_ids[0] == command.members[0].member_id &&
              plan->member_ids[1] == command.members[1].member_id,
          "TP cohort plan records the cohort identity and ordered members");

  const auto c1 = BuildCohortPlan(TpControlCommand{.sequence = 5,
                                                   .max_tokens = 3,
                                                   .prompt_tokens = {7, 8},
                                                   .client_id = "c1"},
                                  &error);
  Require(!c1.has_value() && !error.empty(),
          "TP cohort plan rejects a C1 command");

  auto tampered = MakeC2Command(42, 901);
  tampered.execution_plan_digest = {};
  error.clear();
  Require(!BuildCohortPlan(tampered, &error).has_value() && !error.empty(),
          "TP cohort plan rejects a stale execution plan digest");

  auto zero_cohort = MakeC2Command(43, 902);
  zero_cohort.cohort_id = 0;
  SealCommand(&zero_cohort);
  error.clear();
  Require(!BuildCohortPlan(zero_cohort, &error).has_value() && !error.empty(),
          "TP cohort plan rejects a zero cohort identity");

  auto duplicate = MakeC2Command(44, 903);
  duplicate.members[1].member_id = duplicate.members[0].member_id;
  SealCommand(&duplicate);
  error.clear();
  Require(!BuildCohortPlan(duplicate, &error).has_value() && !error.empty(),
          "TP cohort plan rejects duplicate member identities");

  auto incomplete = MakeC2Command(45, 904);
  incomplete.members.pop_back();
  SealCommand(&incomplete);
  error.clear();
  Require(!BuildCohortPlan(incomplete, &error).has_value() && !error.empty(),
          "TP cohort plan rejects an incomplete cohort");

  auto third_member = MakeC2Command(46, 905);
  third_member.members.push_back({
      .member_id = 906,
      .max_tokens = 2,
      .prompt_tokens = {30},
      .client_id = "member-two",
  });
  SealCommand(&third_member);
  error.clear();
  Require(!BuildCohortPlan(third_member, &error).has_value() && !error.empty(),
          "TP cohort plan rejects a third-member cohort");
}

void TestCohortMembersAreGreedyAndUncached() {
  const auto command = MakeC2Command(51, 950);
  std::string error;
  const auto members = BuildCohortMembers(command, &error);
  Require(members.has_value() && error.empty(),
          "TP cohort members accept a validated C2 command: " + error);
  Require(members->size() == kCohort2MemberCount, "TP cohort has two members");
  for (std::size_t index = 0; index < kCohort2MemberCount; ++index) {
    const auto& member = (*members)[index];
    const auto& source = command.members[index];
    Require(member.member_id == source.member_id &&
                member.prompt ==
                    std::vector<TextRunnerToken>(source.prompt_tokens.begin(),
                                                 source.prompt_tokens.end()) &&
                member.max_tokens == source.max_tokens &&
                member.metadata.client_id == source.client_id,
            "TP cohort member keeps wire identity, prompt and budget");
    Require(member.sampling.can_use_unmodified_argmax(),
            "TP cohort member is greedy");
    Require(!member.publish_token_pieces,
            "TP cohort member does not publish token pieces");
    Require(!static_cast<bool>(member.is_cancelled) &&
                !member.metadata.deadline.has_value() &&
                member.metadata.prompt_context == nullptr,
            "TP cohort member has no cancellation, deadline or continuation");
    Require(!member.metadata.cache_prompt &&
                member.metadata.cache_prefix_tokens == 0,
            "TP cohort member does not reuse prompt state");
  }
  Require((*members)[0].metadata.request_start ==
              (*members)[1].metadata.request_start,
          "TP cohort members share one admission timestamp");

  auto anonymous = MakeC2Command(52, 960);
  anonymous.members[0].client_id.clear();
  SealCommand(&anonymous);
  const auto anonymous_members = BuildCohortMembers(anonymous, &error);
  Require(anonymous_members.has_value() &&
              (*anonymous_members)[0].metadata.client_id == "anonymous",
          "TP cohort member without a client ID uses the anonymous default");
}

void TestCohortMembersRejectInvalidEnvelopes() {
  auto negative_token = MakeC2Command(61, 970);
  negative_token.members[0].prompt_tokens = {10, -1};
  SealCommand(&negative_token);
  RequireRejectedMembers(negative_token,
                         "TP cohort members reject a negative prompt token");

  auto zero_budget = MakeC2Command(62, 971);
  zero_budget.members[1].max_tokens = 0;
  SealCommand(&zero_budget);
  RequireRejectedMembers(zero_budget,
                         "TP cohort members reject a zero token budget");

  auto cached = MakeC2Command(63, 972);
  cached.members[0].cache_prompt = true;
  SealCommand(&cached);
  RequireRejectedMembers(cached, "TP cohort members reject a cached member");

  auto cached_prefix = MakeC2Command(64, 973);
  cached_prefix.members[1].cache_prefix_tokens = 1;
  SealCommand(&cached_prefix);
  RequireRejectedMembers(cached_prefix,
                         "TP cohort members reject a cache prefix");

  auto legacy = MakeC2Command(65, 974);
  legacy.client_id = "legacy-scope";
  SealCommand(&legacy);
  RequireRejectedMembers(legacy,
                         "TP cohort members reject legacy command fields");
}

void TestCohortResponseKeepsMemberOrder() {
  const auto command = MakeC2Command(71, 980);
  std::string error;
  const auto plan = BuildCohortPlan(command, &error);
  Require(plan.has_value(), "TP cohort plan for the response path");
  const auto results = MakeSerialResults();
  const auto translation = BuildCohortResponse(*plan, results, &error);
  Require(translation.response.has_value() && error.empty() &&
              translation.error.empty(),
          "TP cohort response accepts two serial member results: " + error);
  RequireStrictEnvelope(*translation.response, *plan, "response");
  Require((*translation.response).error.empty(),
          "TP cohort success response has no error text");
  Require(
      (*translation.response).members[0].member_id == (*plan).member_ids[0] &&
          (*translation.response).members[1].member_id ==
              (*plan).member_ids[1] &&
          (*translation.response).members[0].tokens ==
              std::vector<std::int32_t>{100, 101} &&
          (*translation.response).members[1].tokens ==
              std::vector<std::int32_t>{200, 201, 202},
      "TP cohort response maps tokens onto ordered member identities");
  for (const auto& member : (*translation.response).members) {
    Require(member.draft_tokens == 0 && member.draft_accepted_tokens == 0 &&
                member.cached_prompt_tokens == 0 &&
                member.cache_snapshot_bytes == 0,
            "TP cohort member result carries no draft or cache telemetry");
  }
}

void TestCohortResponseRejectsNonSerialResults() {
  const auto command = MakeC2Command(81, 990);
  std::string error;
  const auto plan = BuildCohortPlan(command, &error);
  Require(plan.has_value(), "TP cohort plan for the rejection paths");

  auto wide = MakeSerialResults();
  wide[0].physical_execution_width = 2;
  RequireFailureTranslation(BuildCohortResponse(*plan, wide, &error), *plan,
                            "TP cohort response rejects a widened member");

  auto batched = MakeSerialResults();
  batched[1].execution_plan = "cohort-2ar";
  RequireFailureTranslation(BuildCohortResponse(*plan, batched, &error), *plan,
                            "TP cohort response rejects a collective plan");

  auto drafted = MakeSerialResults();
  drafted[0].draft_tokens = 2;
  drafted[0].draft_accepted_tokens = 1;
  RequireFailureTranslation(BuildCohortResponse(*plan, drafted, &error), *plan,
                            "TP cohort response rejects draft telemetry");

  auto cached = MakeSerialResults();
  cached[1].cached_prompt_tokens = 4;
  cached[1].cache_hit = true;
  RequireFailureTranslation(BuildCohortResponse(*plan, cached, &error), *plan,
                            "TP cohort response rejects cached members");

  auto cancelled = MakeSerialResults();
  cancelled[0].cancelled = true;
  RequireFailureTranslation(BuildCohortResponse(*plan, cancelled, &error),
                            *plan,
                            "TP cohort response rejects a cancelled member");

  auto out_of_range = MakeSerialResults();
  out_of_range[1].tokens = {std::numeric_limits<std::uint32_t>::max()};
  RequireFailureTranslation(
      BuildCohortResponse(*plan, out_of_range, &error), *plan,
      "TP cohort response rejects an out-of-range member token");

  auto broken_plan = *plan;
  broken_plan.member_ids[1] = broken_plan.member_ids[0];
  auto duplicate = MakeC2Command(82, 992);
  duplicate.members[1].member_id = duplicate.members[0].member_id;
  SealCommand(&duplicate);
  std::string validation_error;
  Require(!ValidateTpControlCommand(duplicate, &validation_error),
          "TP cohort duplicate member IDs stay invalid on the wire");
  RequireFailureTranslation(
      BuildCohortResponse(broken_plan, MakeSerialResults(), &error),
      broken_plan, "TP cohort response rejects a duplicate plan ID", false);
}

void TestCohortFailureResponseStaysWireValid() {
  const auto command = MakeC2Command(91, 999);
  std::string error;
  const auto plan = BuildCohortPlan(command, &error);
  Require(plan.has_value(), "TP cohort plan for the failure response");

  const auto failure = BuildCohortFailureResponse(*plan, "worker admission");
  RequireStrictEnvelope(failure, *plan, "standalone failure response");
  Require(failure.error == "worker admission" &&
              failure.members[0].tokens.empty() &&
              failure.members[1].tokens.empty(),
          "TP cohort failure response keeps both members and the reason");

  const auto truncated =
      BuildCohortFailureResponse(*plan, std::string((1U << 20) + 64, 'x'));
  RequireStrictEnvelope(truncated, *plan, "truncated failure response");
  Require(truncated.error.size() == (1U << 20),
          "TP cohort failure reason is truncated to the worker byte limit");

  const auto empty = BuildCohortFailureResponse(*plan, std::string_view{});
  Require(empty.error.empty() && empty.members.size() == kCohort2MemberCount,
          "TP cohort failure envelope is valid without a reason");
}

}  // namespace

int main() {
  const auto command = MakeC2Command(1, 100);
  std::string error;
  Require(ValidateTpControlCommand(command, &error),
          "TP cohort test fixture is a valid C2 command: " + error);

  TestCohortPlanAcceptsValidatedCommand();
  TestCohortMembersAreGreedyAndUncached();
  TestCohortMembersRejectInvalidEnvelopes();
  TestCohortResponseKeepsMemberOrder();
  TestCohortResponseRejectsNonSerialResults();
  TestCohortFailureResponseStaysWireValid();
  return 0;
}

#include "src/cli/serve/tp_cohort_plan.hpp"

#include <algorithm>
#include <limits>
#include <utility>

#include "src/cli/serve/text_model_runner.hpp"

namespace gufo::server {
namespace {

using TextCohortResult = TextGenerationBackend::Result;
using TextCohortResults = std::array<TextCohortResult, kCohort2MemberCount>;
using TextCohortMembers =
    std::array<TextCohortMemberRequest, kCohort2MemberCount>;

/// Mirrors the worker protocol member token bound in tp_control.cpp.
constexpr std::size_t kMaxCohortMemberTokens = 1U << 20;
/// Mirrors the worker protocol error byte bound in tp_control.cpp.
constexpr std::size_t kMaxCohortErrorBytes = 1U << 20;
/// The dormant C2 slice still executes each member as independent serial work.
constexpr std::string_view kSerialC1Plan = "serial-c1";
constexpr std::string_view kSerialFallbackPlan = "serial-fallback";

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

void ClearError(std::string* error) {
  if (error != nullptr) {
    error->clear();
  }
}

[[nodiscard]] bool IsZeroDigest(const TpPlanDigest& digest) noexcept {
  return std::ranges::all_of(digest,
                             [](std::uint8_t byte) { return byte == 0; });
}

[[nodiscard]] bool HasValidPlanMemberIds(
    const std::array<std::uint64_t, kCohort2MemberCount>& member_ids) {
  for (std::size_t index = 0; index < member_ids.size(); ++index) {
    if (member_ids[index] == 0) {
      return false;
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (member_ids[previous] == member_ids[index]) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool IsSerialPlanMember(const TextCohortResult& result) {
  return result.physical_execution_width == 1 &&
         (result.execution_plan == kSerialC1Plan ||
          result.execution_plan == kSerialFallbackPlan);
}

[[nodiscard]] bool IsUncachedArMember(const TextCohortResult& result) {
  return result.draft_tokens == 0 && result.draft_accepted_tokens == 0 &&
         result.cached_prompt_tokens == 0 && result.cache_restore_bytes == 0 &&
         result.cache_snapshot_bytes == 0 && result.cache_shared_bytes == 0 &&
         !result.cache_hit && !result.cache_disk_hit;
}

[[nodiscard]] TpControlResponse MakeCohortEnvelope(const TpCohortPlan& plan) {
  TpControlResponse response{
      .sequence = plan.sequence,
      .kind = TpControlResponseKind::kCohort2Ar,
      .cohort_id = plan.cohort_id,
      .execution_plan_digest = plan.execution_plan_digest,
      .cache_plan_digest = plan.cache_plan_digest,
  };
  for (const auto member_id : plan.member_ids) {
    response.members.push_back(TpControlMemberResponse{.member_id = member_id});
  }
  return response;
}

/// Copies a validated member prompt into runner tokens, rejecting any token
/// outside the worker protocol range before it can reach a runner.
[[nodiscard]] bool TranslateMemberPrompt(
    const std::vector<std::int32_t>& prompt_tokens,
    std::vector<TextRunnerToken>* prompt) {
  prompt->clear();
  prompt->reserve(prompt_tokens.size());
  for (const auto token : prompt_tokens) {
    if (token < 0 || static_cast<std::uint64_t>(token) >
                         std::numeric_limits<TextRunnerToken>::max()) {
      return false;
    }
    prompt->push_back(static_cast<TextRunnerToken>(token));
  }
  return true;
}

[[nodiscard]] bool TranslateMemberTokens(const TextCohortResult& result,
                                         TpControlMemberResponse* member) {
  if (result.tokens.size() > kMaxCohortMemberTokens) {
    return false;
  }
  member->tokens.clear();
  member->tokens.reserve(result.tokens.size());
  for (const auto token : result.tokens) {
    if (static_cast<std::uint64_t>(token) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
      return false;
    }
    member->tokens.push_back(static_cast<std::int32_t>(token));
  }
  return true;
}

}  // namespace

std::optional<TpCohortPlan> BuildCohortPlan(const TpControlCommand& command,
                                            std::string* error) {
  ClearError(error);
  if (command.kind != TpControlCommandKind::kCohort2Ar) {
    SetError(error, "TP cohort plan requires a C2 cohort command");
    return std::nullopt;
  }
  if (!ValidateTpControlCommand(command, error)) {
    return std::nullopt;
  }
  TpCohortPlan plan{
      .sequence = command.sequence,
      .cohort_id = command.cohort_id,
      .execution_plan_digest = command.execution_plan_digest,
      .cache_plan_digest = command.cache_plan_digest,
  };
  for (std::size_t index = 0; index < kCohort2MemberCount; ++index) {
    plan.member_ids[index] = command.members[index].member_id;
  }
  return plan;
}

std::optional<TextCohortMembers> BuildCohortMembers(
    const TpControlCommand& command, std::string* error) {
  // An accepted cohort plan is the precondition for building either member.
  if (!BuildCohortPlan(command, error).has_value()) {
    return std::nullopt;
  }

  // One admission timestamp for the whole cohort keeps the members in a single
  // scheduler arrival window; the cohort identity stays plan metadata. The C2
  // envelope carries no sampling fields, so every member translates to the
  // unmodified argmax default and never to a sampling path.
  const auto request_start = TextGenerationScheduler::Clock::now();
  sampling::SamplingConfig greedy;
  greedy.Validate();

  TextCohortMembers members;
  for (std::size_t index = 0; index < kCohort2MemberCount; ++index) {
    const auto& member = command.members[index];
    if (member.max_tokens == 0 ||
        static_cast<std::uint64_t>(member.max_tokens) >
            std::numeric_limits<std::size_t>::max()) {
      SetError(error, "TP cohort member token budget is out of range");
      return std::nullopt;
    }
    if (member.cache_prompt || member.cache_prefix_tokens != 0) {
      SetError(error, "TP cohort members must be uncached");
      return std::nullopt;
    }
    TextCohortMemberRequest request{
        .member_id = member.member_id,
        .max_tokens = static_cast<std::size_t>(member.max_tokens),
        .sampling = greedy,
        .publish_token_pieces = false,
    };
    if (!TranslateMemberPrompt(member.prompt_tokens, &request.prompt)) {
      SetError(error, "TP cohort member prompt token is out of range");
      return std::nullopt;
    }
    request.metadata = TextRequestMetadata{
        .client_id = member.client_id.empty() ? "anonymous" : member.client_id,
        .deadline = std::nullopt,
        .request_start = request_start,
        .prompt_context = nullptr,
        .cache_prompt = false,
        .cache_prefix_tokens = 0,
    };
    members[index] = std::move(request);
  }
  return members;
}

TpCohortTranslation BuildCohortResponse(const TpCohortPlan& plan,
                                        const TextCohortResults& results,
                                        std::string* error) {
  TpCohortTranslation translation{
      .response = std::nullopt,
      .failure = BuildCohortFailureResponse(plan, std::string_view{}),
      .error = {},
  };
  // Every rejection keeps the full two-member envelope, so a failed cohort is
  // never reported as a truncated C2 response.
  const auto fail = [&translation, &plan, error](std::string reason) {
    translation.error = std::move(reason);
    translation.failure = BuildCohortFailureResponse(plan, translation.error);
    if (error != nullptr) {
      *error = translation.error;
    }
    return translation;
  };

  if (plan.cohort_id == 0 || IsZeroDigest(plan.execution_plan_digest) ||
      IsZeroDigest(plan.cache_plan_digest) ||
      !HasValidPlanMemberIds(plan.member_ids)) {
    return fail("TP cohort response requires an accepted cohort plan");
  }

  auto response = MakeCohortEnvelope(plan);
  for (std::size_t index = 0; index < kCohort2MemberCount; ++index) {
    const auto& result = results[index];
    if (!IsSerialPlanMember(result)) {
      return fail(
          "TP cohort members must report serial width and a serial "
          "execution plan");
    }
    if (result.cancelled) {
      return fail("TP cohort member request was cancelled");
    }
    if (!IsUncachedArMember(result)) {
      return fail("TP cohort response requires uncached AR member results");
    }
    if (!TranslateMemberTokens(result, &response.members[index])) {
      return fail("TP cohort member produced an out-of-range token");
    }
  }

  if (!ValidateTpControlResponse(response, &translation.error)) {
    return fail(translation.error);
  }
  ClearError(error);
  translation.response = std::move(response);
  return translation;
}

TpControlResponse BuildCohortFailureResponse(const TpCohortPlan& plan,
                                             std::string_view reason) {
  auto response = MakeCohortEnvelope(plan);
  if (reason.size() > kMaxCohortErrorBytes) {
    reason = reason.substr(0, kMaxCohortErrorBytes);
  }
  response.error = reason;
  return response;
}

}  // namespace gufo::server

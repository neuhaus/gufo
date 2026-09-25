#ifndef GUFO_SERVER_TP_COHORT_PLAN_HPP_
#define GUFO_SERVER_TP_COHORT_PLAN_HPP_

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/cli/serve/tp_control.hpp"

namespace gufo::server {

/// Admission metadata for one accepted C2 cohort.
///
/// `cohort_id` is the future shared C2 scope and is recorded as metadata only:
/// this seam never opens, binds or executes a collective under it, and members
/// still run as independent serial C1 work. The digests and the ordered member
/// IDs are echoed verbatim by both response builders.
struct TpCohortPlan {
  std::uint64_t sequence{0};
  std::uint64_t cohort_id{0};
  TpPlanDigest execution_plan_digest{};
  TpPlanDigest cache_plan_digest{};
  std::array<std::uint64_t, kCohort2MemberCount> member_ids{};
};

/// Accepted member translation, a strict response value, and one wire-valid
/// failure value. The cohort identity is API metadata only: nothing here
/// selects, binds or executes a shared C2 collective.
struct TpCohortTranslation {
  std::optional<TpControlResponse> response;
  TpControlResponse failure;
  std::string error;
};

/// Validates one `kCohort2Ar` command and returns its admission plan.
///
/// A C1 command, an incomplete or over-wide cohort, duplicate or zero member
/// IDs, and any envelope or plan-digest mismatch are rejected before a caller
/// can build members from the command.
[[nodiscard]] std::optional<TpCohortPlan> BuildCohortPlan(
    const TpControlCommand& command, std::string* error);

/// Translates a validated C2 command into exactly two ordered scheduler
/// admission members.
///
/// The translation is greedy, non-streaming, uncached, and deadline-free, with
/// no cancellation callback and no prompt continuation. It selects no
/// distributed runner plan, and the command cohort identity stays metadata for
/// the caller; no collective scope is opened here.
[[nodiscard]] std::optional<
    std::array<TextCohortMemberRequest, kCohort2MemberCount>>
BuildCohortMembers(const TpControlCommand& command, std::string* error);

/// Translates two member results into one strict C2 response.
///
/// Members are matched to plan member IDs by position, so the response keeps
/// command order. Every member must report serial width, a serial execution
/// plan, and zero draft, cache and cancellation telemetry; a token outside the
/// worker protocol range also fails. A failure is returned as a wire-valid C2
/// error response that still carries both members, so a rejected cohort is
/// never truncated to a partial envelope.
[[nodiscard]] TpCohortTranslation BuildCohortResponse(
    const TpCohortPlan& plan,
    const std::array<TextGenerationBackend::Result, kCohort2MemberCount>&
        results,
    std::string* error);

/// Truncates one failure reason to the worker error byte limit and returns the
/// wire-valid C2 error response for an accepted plan. Both member envelopes
/// are present with empty token lists.
[[nodiscard]] TpControlResponse BuildCohortFailureResponse(
    const TpCohortPlan& plan, std::string_view reason);

}  // namespace gufo::server

#endif  // GUFO_SERVER_TP_COHORT_PLAN_HPP_

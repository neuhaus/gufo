#include "src/cli/serve/tp_control.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using gufo::server::ComputeTpCachePlanDigest;
using gufo::server::ComputeTpExecutionPlanDigest;
using gufo::server::TpControlChannel;
using gufo::server::TpControlCommand;
using gufo::server::TpControlCommandKind;
using gufo::server::TpControlConfig;
using gufo::server::TpControlResponse;
using gufo::server::TpControlResponseKind;
using gufo::server::TpInstructionOp;
using gufo::server::TpPlanDigest;
using gufo::server::TpResponseBroker;
using gufo::server::TpResponseExpectation;
using gufo::server::ValidateTpControlCommand;
using gufo::server::ValidateTpControlResponse;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    std::exit(1);
  }
}

std::uint16_t FreePort() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    std::perror("socket");
    std::exit(1);
  }
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    std::perror("bind");
    ::close(fd);
    std::exit(1);
  }
  socklen_t length = sizeof(address);
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
    std::perror("getsockname");
    ::close(fd);
    std::exit(1);
  }
  const auto port = ntohs(address.sin_port);
  ::close(fd);
  return port;
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
  command.execution_plan_digest = ComputeTpExecutionPlanDigest(command);
  command.cache_plan_digest = ComputeTpCachePlanDigest(command);
  return command;
}

TpControlResponse MakeC2Response(const TpControlCommand& command) {
  return {
      .sequence = command.sequence,
      .kind = TpControlResponseKind::kCohort2Ar,
      .cohort_id = command.cohort_id,
      .execution_plan_digest = command.execution_plan_digest,
      .cache_plan_digest = command.cache_plan_digest,
      .members =
          {
              {
                  .member_id = command.members[0].member_id,
                  .tokens = {100, 101},
              },
              {
                  .member_id = command.members[1].member_id,
                  .tokens = {200, 201, 202},
              },
          },
  };
}

TpResponseExpectation MakeC2Expectation(const TpControlCommand& command) {
  return {
      .cohort_id = command.cohort_id,
      .execution_plan_digest = command.execution_plan_digest,
      .cache_plan_digest = command.cache_plan_digest,
      .member_ids = {command.members[0].member_id,
                     command.members[1].member_id},
  };
}

void RequireC2MembersInOrder(const TpControlCommand& command,
                             const TpControlResponse& response) {
  Require(
      response.sequence == command.sequence &&
          response.kind == TpControlResponseKind::kCohort2Ar &&
          response.cohort_id == command.cohort_id &&
          response.execution_plan_digest == command.execution_plan_digest &&
          response.cache_plan_digest == command.cache_plan_digest &&
          response.members.size() == command.members.size() &&
          response.members[0].member_id == command.members[0].member_id &&
          response.members[1].member_id == command.members[1].member_id &&
          response.members[0].tokens == std::vector<std::int32_t>{100, 101} &&
          response.members[1].tokens ==
              std::vector<std::int32_t>{200, 201, 202},
      "TP C2 response preserves cohort and member order");
}

void RequireInvalidCommand(const TpControlCommand& command,
                           const std::string& message) {
  std::string error;
  Require(!ValidateTpControlCommand(command, &error) && !error.empty(),
          message);
}

void RequireInvalidResponse(const TpControlResponse& response,
                            const std::string& message) {
  std::string error;
  Require(!ValidateTpControlResponse(response, &error) && !error.empty(),
          message);
}

}  // namespace

int main() {
  const auto port = FreePort();
  std::string client_error;
  std::shared_ptr<TpControlChannel> client;
  std::thread connector([&] {
    client = TpControlChannel::Connect("127.0.0.1", port, &client_error);
  });

  std::string server_error;
  auto server = TpControlChannel::Listen(port, &server_error);
  connector.join();
  Require(server != nullptr, server_error);
  Require(client != nullptr, client_error);

  TpControlConfig rank0{.snapshot_budget_bytes = 8192,
                        .rank = 0,
                        .world_size = 2,
                        .max_context = 4096,
                        .max_draft_tokens = 7,
                        .use_mtp = true,
                        .allow_cache_reuse = true,
                        .auth_token = "test-token",
                        .prefill_chunk_tokens = 512};
  TpControlConfig rank1 = rank0;
  rank1.rank = 1;
  rank1.snapshot_budget_bytes = 4096;
  bool server_handshake = false;
  bool client_handshake = false;
  std::thread server_handshake_thread(
      [&] { server_handshake = server->Handshake(rank0, &server_error); });
  std::thread client_handshake_thread(
      [&] { client_handshake = client->Handshake(rank1, &client_error); });
  server_handshake_thread.join();
  client_handshake_thread.join();
  Require(server_handshake, server_error);
  Require(client_handshake, client_error);
  Require(server->snapshot_budget_bytes() == 4096 &&
              client->snapshot_budget_bytes() == 4096,
          "cache budget is the smaller host budget");

  TpControlCommand command{.sequence = 7,
                           .max_tokens = 4,
                           .cache_prompt = true,
                           .cache_prefix_tokens = 2,
                           .prompt_tokens = {10, 11, 12},
                           .client_id = "probe"};
  Require(server->SendCommand(command, &server_error), server_error);
  TpControlCommand received;
  Require(client->ReceiveCommand(&received, &client_error), client_error);
  Require(received.sequence == command.sequence &&
              received.kind == TpControlCommandKind::kSingle &&
              received.cohort_id == command.sequence &&
              received.members.empty() &&
              received.max_tokens == command.max_tokens &&
              received.cache_prompt == command.cache_prompt &&
              received.cache_prefix_tokens == command.cache_prefix_tokens &&
              received.prompt_tokens == command.prompt_tokens &&
              received.client_id == command.client_id,
          "TP control command round trip");

  // A C1 command carries the request's whole sampling configuration: rank 1
  // builds the same sampler for multi-token decoding, bit for bit.
  {
    TpControlCommand sampled = command;
    sampled.sequence = 8;
    sampled.sampling = {.temperature = 0.8F,
                        .top_k = 40,
                        .top_p = 0.9F,
                        .min_p = 0.05F,
                        .min_keep = 2,
                        .seed = 7,
                        .repeat_penalty = 1.1F,
                        .repeat_last_n = 32,
                        .frequency_penalty = 0.2F,
                        .presence_penalty = -0.3F};
    Require(server->SendCommand(sampled, &server_error), server_error);
    TpControlCommand got;
    Require(client->ReceiveCommand(&got, &client_error), client_error);
    const auto& want = sampled.sampling;
    Require(got.sampling.temperature == want.temperature &&
                got.sampling.top_k == want.top_k &&
                got.sampling.top_p == want.top_p &&
                got.sampling.min_p == want.min_p &&
                got.sampling.min_keep == want.min_keep &&
                got.sampling.seed == want.seed &&
                got.sampling.repeat_penalty == want.repeat_penalty &&
                got.sampling.repeat_last_n == want.repeat_last_n &&
                got.sampling.frequency_penalty == want.frequency_penalty &&
                got.sampling.presence_penalty == want.presence_penalty &&
                received.sampling.can_use_unmodified_argmax(),
            "TP C1 command preserves the sampling configuration");
    TpControlCommand invalid = sampled;
    invalid.sampling.top_p = 0.0F;
    std::string why;
    Require(!server->SendCommand(invalid, &why) && !why.empty(),
            "TP C1 command with invalid sampling is refused");
  }

  // An instruction is one model call for rank 1 to execute within the request
  // named by `sequence`. Every argument must survive the round trip exactly,
  // and a negative token must survive by bit pattern for validation to see it.
  const auto round_trip = [&](const TpControlCommand& sent) {
    std::string why;
    Require(ValidateTpControlCommand(sent, &why), why);
    Require(server->SendCommand(sent, &server_error), server_error);
    TpControlCommand got;
    Require(client->ReceiveCommand(&got, &client_error), client_error);
    Require(got.sequence == sent.sequence &&
                got.kind == TpControlCommandKind::kInstruction &&
                got.members.empty() && got.prompt_tokens.empty() &&
                got.instruction == sent.instruction,
            "TP instruction round trip");
  };
  const TpControlCommand prefill{
      .sequence = command.sequence,
      .kind = TpControlCommandKind::kInstruction,
      .instruction = {.op = TpInstructionOp::kPrefill,
                      .index = 9,
                      .state = 1,
                      .offset = 512,
                      .count = 512,
                      .prompt_size = 2054}};
  round_trip(prefill);
  round_trip(
      {.sequence = command.sequence,
       .kind = TpControlCommandKind::kInstruction,
       .instruction = {
           .op = TpInstructionOp::kAdvance, .index = 10, .token = 248068}});
  round_trip({.sequence = command.sequence,
              .kind = TpControlCommandKind::kInstruction,
              .instruction = {.op = TpInstructionOp::kDecode,
                              .index = 11,
                              .count = 8,
                              .rng = 0x0123456789abcdefULL,
                              .pending = 248067}});
  round_trip({.sequence = command.sequence,
              .kind = TpControlCommandKind::kInstruction,
              .instruction = {.op = TpInstructionOp::kEnd,
                              .index = 12,
                              .count = 3,
                              .digest = 0xfedcba9876543210ULL}});
  // A reset is the one call the continuation cache can make between requests.
  round_trip(
      {.sequence = 0,
       .kind = TpControlCommandKind::kInstruction,
       .instruction = {
           .op = TpInstructionOp::kInvalidate, .index = 13, .state = 0}});

  for (const auto op : {TpInstructionOp::kSnapshot, TpInstructionOp::kRestore,
                        TpInstructionOp::kDrop}) {
    round_trip(
        {.sequence = command.sequence,
         .kind = TpControlCommandKind::kInstruction,
         .instruction = {
             .op = op, .index = 14, .snapshot_id = 0xffffffffffffffffULL}});
  }
  round_trip(
      {.sequence = 0,
       .kind = TpControlCommandKind::kInstruction,
       .instruction = {
           .op = TpInstructionOp::kDrop, .index = 15, .snapshot_id = 19}});
  round_trip(
      {.sequence = command.sequence,
       .kind = TpControlCommandKind::kInstruction,
       .instruction = {
           .op = TpInstructionOp::kReuse, .index = 16, .prompt_size = 2048}});
  round_trip(
      {.sequence = command.sequence,
       .kind = TpControlCommandKind::kInstruction,
       .instruction = {.op = TpInstructionOp::kCancelPrepare, .index = 17}});
  const TpControlResponse ack{.instruction_index = 17,
                              .sequence = command.sequence,
                              .error = "capture failed",
                              .kind = TpControlResponseKind::kInstruction};
  Require(client->SendResponse(ack, &client_error), client_error);
  TpControlResponse got_ack;
  Require(server->ReceiveResponse(&got_ack, &server_error), server_error);
  Require(got_ack.kind == ack.kind && got_ack.sequence == ack.sequence &&
              got_ack.instruction_index == ack.instruction_index &&
              got_ack.error == ack.error,
          "cache acknowledgement round trip");

  // A malformed instruction must be refused by the validator and on the wire,
  // because the wire is the path a peer exercises.
  TpPlanDigest nonzero_digest{};
  nonzero_digest[0] = 1;
  const auto refuses = [&](const std::string& what,
                           const TpControlCommand& bad) {
    std::string why;
    Require(
        !ValidateTpControlCommand(bad, &why) && !why.empty(),
        "TP instruction validation must refuse " + what + ", but said: " + why);
    std::string send_why;
    Require(
        !server->SendCommand(bad, &send_why) && !send_why.empty(),
        "TP instruction send must refuse " + what + ", but said: " + send_why);
  };
  const auto with = [&](auto change) {
    TpControlCommand bad = prefill;
    change(bad);
    return bad;
  };
  // Fields that belong to the request, not to one of its calls.
  refuses("a member", with([&](auto& bad) {
            bad.members = {{.member_id = command.sequence, .max_tokens = 4}};
          }));
  refuses("a prompt", with([](auto& bad) { bad.prompt_tokens = {10, 11}; }));
  refuses("a token budget", with([](auto& bad) { bad.max_tokens = 4; }));
  refuses("a cache request", with([](auto& bad) { bad.cache_prompt = true; }));
  refuses("a cache prefix",
          with([](auto& bad) { bad.cache_prefix_tokens = 2; }));
  refuses("a client id", with([](auto& bad) { bad.client_id = "probe"; }));
  refuses("a sampling configuration",
          with([](auto& bad) { bad.sampling.temperature = 0.5F; }));
  refuses("a draw state outside a decode",
          with([](auto& bad) { bad.instruction.rng = 1; }));
  refuses("a pending draw outside a decode",
          with([](auto& bad) { bad.instruction.pending = 3; }));
  refuses("a cohort scope", with([](auto& bad) { bad.cohort_id = 3; }));
  refuses("an execution-plan digest",
          with([&](auto& bad) { bad.execution_plan_digest = nonzero_digest; }));
  refuses("a cache-plan digest",
          with([&](auto& bad) { bad.cache_plan_digest = nonzero_digest; }));
  // Arguments that do not fit the operation.
  refuses("an empty prefill",
          with([](auto& bad) { bad.instruction.count = 0; }));
  refuses("a prefill past its prompt",
          with([](auto& bad) { bad.instruction.prompt_size = 512; }));
  refuses("a prefill carrying a token",
          with([](auto& bad) { bad.instruction.token = 5; }));
  refuses("a negative token", with([](auto& bad) {
            bad.instruction = {.op = TpInstructionOp::kAdvance, .token = -7};
          }));
  refuses("a one-token decode", with([](auto& bad) {
            bad.instruction = {.op = TpInstructionOp::kDecode, .count = 1};
          }));
  refuses("a snapshot with ID zero", with([](auto& bad) {
            bad.instruction = {.op = TpInstructionOp::kSnapshot};
          }));
  refuses("a stray snapshot ID",
          with([](auto& bad) { bad.instruction.snapshot_id = 1; }));
  refuses("idle restore", with([](auto& bad) {
            bad.sequence = 0;
            bad.instruction = {.op = TpInstructionOp::kRestore,
                               .snapshot_id = 1};
          }));
  refuses("a missing operation", with([](auto& bad) { bad.instruction = {}; }));
  refuses("a model call outside a request",
          with([](auto& bad) { bad.sequence = 0; }));
  {
    TpControlCommand bad = command;
    bad.instruction = {.op = TpInstructionOp::kAdvance, .token = 1};
    refuses("an instruction on a request command", bad);
  }

  TpControlResponse response{.sequence = received.sequence,
                             .tokens = {20, 21},
                             .draft_tokens = 3,
                             .draft_accepted_tokens = 2,
                             .cached_prompt_tokens = 2,
                             .cache_snapshot_bytes = 4096,
                             .error = {}};
  Require(client->SendResponse(response, &client_error), client_error);
  TpControlResponse received_response;
  Require(server->ReceiveResponse(&received_response, &server_error),
          server_error);
  Require(received_response.sequence == response.sequence &&
              received_response.kind == TpControlResponseKind::kSingle &&
              received_response.cohort_id == response.sequence &&
              received_response.members.empty() &&
              received_response.tokens == response.tokens &&
              received_response.draft_tokens == response.draft_tokens &&
              received_response.draft_accepted_tokens ==
                  response.draft_accepted_tokens &&
              received_response.cached_prompt_tokens ==
                  response.cached_prompt_tokens &&
              received_response.cache_snapshot_bytes ==
                  response.cache_snapshot_bytes,
          "TP control response round trip");

  TpControlCommand zero_scope_command{
      .sequence = 0,
      .max_tokens = 1,
      .cache_prompt = false,
      .cache_prefix_tokens = 0,
      .prompt_tokens = {13},
      .client_id = "zero-scope",
  };
  TpControlCommand received_zero_scope;
  Require(server->SendCommand(zero_scope_command, &server_error), server_error);
  Require(client->ReceiveCommand(&received_zero_scope, &client_error),
          client_error);
  Require(received_zero_scope.sequence == 0,
          "TP control preserves a zero operation scope");
  const TpControlResponse zero_scope_response{
      .sequence = 0, .tokens = {14}, .error = {}};
  TpControlResponse received_zero_scope_response;
  Require(client->SendResponse(zero_scope_response, &client_error),
          client_error);
  Require(server->ReceiveResponse(&received_zero_scope_response, &server_error),
          server_error);
  Require(
      received_zero_scope_response.sequence == 0 &&
          received_zero_scope_response.kind == TpControlResponseKind::kSingle &&
          received_zero_scope_response.cohort_id == 0,
      "TP control response preserves a zero operation scope");

  const auto c2_command = MakeC2Command(20, 1000);
  std::string validation_error;
  Require(ValidateTpControlCommand(c2_command, &validation_error),
          validation_error);
  Require(
      ComputeTpExecutionPlanDigest(c2_command) ==
              c2_command.execution_plan_digest &&
          ComputeTpCachePlanDigest(c2_command) == c2_command.cache_plan_digest,
      "TP C2 digest helpers are deterministic");

  auto reordered = c2_command;
  std::swap(reordered.members[0], reordered.members[1]);
  Require(
      ComputeTpExecutionPlanDigest(reordered) !=
              c2_command.execution_plan_digest &&
          ComputeTpCachePlanDigest(reordered) != c2_command.cache_plan_digest,
      "TP C2 digests bind member order");
  auto changed_prompt = c2_command;
  changed_prompt.members[0].prompt_tokens.push_back(13);
  Require(ComputeTpExecutionPlanDigest(changed_prompt) !=
                  c2_command.execution_plan_digest &&
              ComputeTpCachePlanDigest(changed_prompt) ==
                  c2_command.cache_plan_digest,
          "TP execution digest binds prompts without changing the cache plan");

  auto malformed = c2_command;
  malformed.members.pop_back();
  malformed.execution_plan_digest = ComputeTpExecutionPlanDigest(malformed);
  malformed.cache_plan_digest = ComputeTpCachePlanDigest(malformed);
  RequireInvalidCommand(malformed, "TP C2 rejects a one-member cohort");
  malformed = c2_command;
  malformed.members[1].member_id = malformed.members[0].member_id;
  malformed.execution_plan_digest = ComputeTpExecutionPlanDigest(malformed);
  malformed.cache_plan_digest = ComputeTpCachePlanDigest(malformed);
  RequireInvalidCommand(malformed, "TP C2 rejects duplicate member IDs");
  malformed = c2_command;
  malformed.members.push_back({
      .member_id = c2_command.cohort_id + 3,
      .max_tokens = 1,
      .prompt_tokens = {30},
      .client_id = "member-extra",
  });
  malformed.execution_plan_digest = ComputeTpExecutionPlanDigest(malformed);
  malformed.cache_plan_digest = ComputeTpCachePlanDigest(malformed);
  RequireInvalidCommand(malformed, "TP C2 rejects more than two members");
  malformed = c2_command;
  malformed.members[1].cache_prompt = true;
  malformed.execution_plan_digest = ComputeTpExecutionPlanDigest(malformed);
  malformed.cache_plan_digest = ComputeTpCachePlanDigest(malformed);
  RequireInvalidCommand(malformed, "TP C2 rejects cache-enabled members");
  malformed = c2_command;
  malformed.execution_plan_digest[0] ^= 1U;
  RequireInvalidCommand(malformed, "TP C2 rejects an execution-plan mismatch");
  malformed = c2_command;
  malformed.cache_plan_digest[31] ^= 1U;
  RequireInvalidCommand(malformed, "TP C2 rejects a cache-plan mismatch");

  const auto c2_response = MakeC2Response(c2_command);
  validation_error.clear();
  Require(ValidateTpControlResponse(c2_response, &validation_error),
          validation_error);
  auto malformed_response = c2_response;
  malformed_response.members.pop_back();
  RequireInvalidResponse(malformed_response,
                         "TP C2 rejects a one-member response");
  malformed_response = c2_response;
  malformed_response.members[1].member_id =
      malformed_response.members[0].member_id;
  RequireInvalidResponse(malformed_response,
                         "TP C2 rejects duplicate response member IDs");
  malformed_response = c2_response;
  malformed_response.members[0].draft_tokens = 1;
  RequireInvalidResponse(malformed_response,
                         "TP C2 rejects an AR response with draft tokens");
  malformed_response = c2_response;
  malformed_response.members[1].cached_prompt_tokens = 1;
  RequireInvalidResponse(malformed_response,
                         "TP C2 rejects a cached C2 response");
  malformed_response = c2_response;
  malformed_response.execution_plan_digest = {};
  RequireInvalidResponse(malformed_response,
                         "TP C2 rejects a missing execution digest");

  auto bad_wire_command = c2_command;
  bad_wire_command.execution_plan_digest[0] ^= 1U;
  server_error.clear();
  Require(!server->SendCommand(bad_wire_command, &server_error) &&
              server_error.find("digest mismatch") != std::string::npos,
          "TP channel rejects a C2 digest mismatch before sending");
  Require(server->SendCommand(c2_command, &server_error), server_error);
  TpControlCommand received_c2_command;
  Require(client->ReceiveCommand(&received_c2_command, &client_error),
          client_error);
  Require(received_c2_command.sequence == c2_command.sequence &&
              received_c2_command.kind == TpControlCommandKind::kCohort2Ar &&
              received_c2_command.cohort_id == c2_command.cohort_id &&
              received_c2_command.execution_plan_digest ==
                  c2_command.execution_plan_digest &&
              received_c2_command.cache_plan_digest ==
                  c2_command.cache_plan_digest &&
              received_c2_command.members.size() == 2 &&
              received_c2_command.members[0].member_id ==
                  c2_command.members[0].member_id &&
              received_c2_command.members[1].member_id ==
                  c2_command.members[1].member_id &&
              received_c2_command.members[0].prompt_tokens ==
                  c2_command.members[0].prompt_tokens &&
              received_c2_command.members[1].prompt_tokens ==
                  c2_command.members[1].prompt_tokens,
          "TP C2 command round trip preserves the ordered member plan");
  Require(client->SendResponse(c2_response, &client_error), client_error);
  TpControlResponse received_c2_response;
  Require(server->ReceiveResponse(&received_c2_response, &server_error),
          server_error);
  RequireC2MembersInOrder(c2_command, received_c2_response);

  const auto mismatch_port = FreePort();
  std::shared_ptr<TpControlChannel> mismatch_client;
  std::thread mismatch_connector([&] {
    mismatch_client =
        TpControlChannel::Connect("127.0.0.1", mismatch_port, &client_error);
  });
  auto mismatch_server = TpControlChannel::Listen(mismatch_port, &server_error);
  mismatch_connector.join();
  Require(mismatch_server != nullptr && mismatch_client != nullptr,
          "TP control mismatch pair connects");
  TpControlConfig mismatch_rank0 = rank0;
  TpControlConfig mismatch_rank1 = rank1;
  mismatch_rank1.allow_cache_reuse = false;
  bool mismatch_server_handshake = false;
  bool mismatch_client_handshake = false;
  std::thread mismatch_server_thread([&] {
    mismatch_server_handshake =
        mismatch_server->Handshake(mismatch_rank0, &server_error);
  });
  std::thread mismatch_client_thread([&] {
    mismatch_client_handshake =
        mismatch_client->Handshake(mismatch_rank1, &client_error);
  });
  mismatch_server_thread.join();
  mismatch_client_thread.join();
  Require(!mismatch_server_handshake && !mismatch_client_handshake,
          "TP control rejects cache-policy mismatch");

  const auto broker_port = FreePort();
  std::shared_ptr<TpControlChannel> broker_client;
  std::thread broker_connector([&] {
    broker_client =
        TpControlChannel::Connect("127.0.0.1", broker_port, &client_error);
  });
  auto broker_server = TpControlChannel::Listen(broker_port, &server_error);
  broker_connector.join();
  Require(broker_server != nullptr && broker_client != nullptr,
          "TP broker pair connects");
  bool broker_server_handshake = false;
  bool broker_client_handshake = false;
  std::thread broker_server_thread([&] {
    broker_server_handshake = broker_server->Handshake(rank0, &server_error);
  });
  std::thread broker_client_thread([&] {
    broker_client_handshake = broker_client->Handshake(rank1, &client_error);
  });
  broker_server_thread.join();
  broker_client_thread.join();
  Require(broker_server_handshake && broker_client_handshake,
          "TP broker pair handshakes");

  TpResponseBroker broker(broker_server, 3);
  Require(broker.RegisterPendingResponse(7, &server_error), server_error);
  Require(broker.RegisterPendingResponse(9, MakeC2Expectation(c2_command),
                                         &server_error),
          server_error);
  Require(broker.RegisterPendingResponse(11, &server_error), server_error);

  TpControlCommand broker_command{.sequence = 11,
                                  .max_tokens = 1,
                                  .cache_prompt = false,
                                  .cache_prefix_tokens = 0,
                                  .prompt_tokens = {42},
                                  .client_id = "broker-command"};
  TpControlCommand received_command;
  bool command_received = false;
  std::thread command_thread([&] {
    command_received =
        broker_client->ReceiveCommand(&received_command, &client_error);
  });
  const bool command_sent =
      broker_server->SendCommand(broker_command, &server_error);
  command_thread.join();
  Require(command_sent && command_received &&
              received_command.sequence == broker_command.sequence,
          "TP broker permits full-duplex command sending");

  bool responses_sent = false;
  std::thread response_thread([&] {
    auto nine = c2_response;
    nine.sequence = 9;
    const TpControlResponse seven{.sequence = 7, .tokens = {17}};
    const TpControlResponse eleven{.sequence = 11, .tokens = {11}};
    responses_sent = broker_client->SendResponse(nine, &client_error) &&
                     broker_client->SendResponse(seven, &client_error) &&
                     broker_client->SendResponse(eleven, &client_error);
  });
  TpControlResponse seven_response;
  TpControlResponse nine_response;
  TpControlResponse eleven_response;
  const bool seven_ready =
      broker.WaitForResponse(7, &seven_response, &server_error);
  const bool nine_ready =
      broker.WaitForResponse(9, &nine_response, &server_error);
  const bool eleven_ready =
      broker.WaitForResponse(11, &eleven_response, &server_error);
  response_thread.join();
  Require(responses_sent && seven_ready && nine_ready && eleven_ready &&
              seven_response.tokens == std::vector<std::int32_t>{17} &&
              eleven_response.tokens == std::vector<std::int32_t>{11},
          "TP broker routes C1 and C2 out-of-order responses by sequence");
  auto broker_c2_command = c2_command;
  broker_c2_command.sequence = 9;
  RequireC2MembersInOrder(broker_c2_command, nine_response);

  Require(broker.RegisterPendingResponse(12, &server_error), server_error);
  const TpControlResponse unexpected{.sequence = 13, .tokens = {130}};
  Require(broker_client->SendResponse(unexpected, &client_error), client_error);
  TpControlResponse ignored;
  Require(!broker.WaitForResponse(12, &ignored, &server_error) &&
              server_error.find("unknown or duplicate") != std::string::npos,
          "TP broker fails closed on an unknown response");
  Require(!broker.RegisterPendingResponse(14, &server_error),
          "TP broker rejects registrations after poisoning");

  const auto contract_mismatch_port = FreePort();
  std::shared_ptr<TpControlChannel> contract_mismatch_client;
  std::thread contract_connector([&] {
    contract_mismatch_client = TpControlChannel::Connect(
        "127.0.0.1", contract_mismatch_port, &client_error);
  });
  auto contract_mismatch_server =
      TpControlChannel::Listen(contract_mismatch_port, &server_error);
  contract_connector.join();
  Require(contract_mismatch_server != nullptr &&
              contract_mismatch_client != nullptr,
          "TP contract mismatch pair connects");
  bool contract_server_handshake = false;
  bool contract_client_handshake = false;
  std::thread contract_server_thread([&] {
    contract_server_handshake =
        contract_mismatch_server->Handshake(rank0, &server_error);
  });
  std::thread contract_client_thread([&] {
    contract_client_handshake =
        contract_mismatch_client->Handshake(rank1, &client_error);
  });
  contract_server_thread.join();
  contract_client_thread.join();
  Require(contract_server_handshake && contract_client_handshake,
          "TP contract mismatch pair handshakes");

  TpResponseBroker contract_broker(contract_mismatch_server, 1);
  auto wrong_expectation = MakeC2Expectation(c2_command);
  std::swap(wrong_expectation.member_ids[0], wrong_expectation.member_ids[1]);
  Require(contract_broker.RegisterPendingResponse(31, wrong_expectation,
                                                  &server_error),
          server_error);
  auto mismatched_response = c2_response;
  mismatched_response.sequence = 31;
  Require(contract_mismatch_client->SendResponse(mismatched_response,
                                                 &client_error),
          client_error);
  TpControlResponse ignored_contract;
  Require(
      !contract_broker.WaitForResponse(31, &ignored_contract, &server_error) &&
          server_error.find("cohort contract mismatch") != std::string::npos,
      "TP broker rejects a C2 member-order mismatch");
  Require(!contract_broker.RegisterPendingResponse(32, &server_error),
          "TP broker remains poisoned after a C2 contract mismatch");

  const auto interrupt_port = FreePort();
  std::shared_ptr<TpControlChannel> interrupt_client;
  std::thread interrupt_connector([&] {
    interrupt_client =
        TpControlChannel::Connect("127.0.0.1", interrupt_port, &client_error);
  });
  auto interrupt_server =
      TpControlChannel::Listen(interrupt_port, &server_error);
  interrupt_connector.join();
  Require(interrupt_server != nullptr && interrupt_client != nullptr,
          "TP interrupt pair connects");
  bool interrupt_server_handshake = false;
  bool interrupt_client_handshake = false;
  std::thread interrupt_server_thread([&] {
    interrupt_server_handshake =
        interrupt_server->Handshake(rank0, &server_error);
  });
  std::thread interrupt_client_thread([&] {
    interrupt_client_handshake =
        interrupt_client->Handshake(rank1, &client_error);
  });
  interrupt_server_thread.join();
  interrupt_client_thread.join();
  Require(interrupt_server_handshake && interrupt_client_handshake,
          "TP interrupt pair handshakes");

  TpResponseBroker interrupt_broker(interrupt_server, 1);
  Require(interrupt_broker.RegisterPendingResponse(21, &server_error),
          server_error);
  bool interrupt_wait_returned = false;
  bool interrupt_wait_ok = false;
  std::string interrupt_wait_error;
  std::thread interrupt_waiter([&] {
    TpControlResponse response;
    interrupt_wait_ok =
        interrupt_broker.WaitForResponse(21, &response, &interrupt_wait_error);
    interrupt_wait_returned = true;
  });
  interrupt_broker.FailAll("test response reader interruption");
  interrupt_waiter.join();
  Require(interrupt_wait_returned && !interrupt_wait_ok &&
              interrupt_wait_error.find("test response reader interruption") !=
                  std::string::npos,
          "TP broker failure wakes a blocked response waiter");

  // An idle pair must outlive its I/O timeout: after the handshake the worker
  // waits for its next command, and rank 0's broker for its next response, for
  // as long as the server is idle. A short timeout stands in for the
  // production default, which used to end an idle pair after 24 hours.
  constexpr std::chrono::milliseconds kShortIoTimeout{300};
  const auto idle_port = FreePort();
  std::shared_ptr<TpControlChannel> idle_client;
  std::thread idle_connector([&] {
    idle_client = TpControlChannel::Connect("127.0.0.1", idle_port,
                                            &client_error, kShortIoTimeout);
  });
  auto idle_server =
      TpControlChannel::Listen(idle_port, &server_error, kShortIoTimeout);
  idle_connector.join();
  Require(idle_server != nullptr && idle_client != nullptr,
          "TP idle pair connects");
  bool idle_server_handshake = false;
  bool idle_client_handshake = false;
  std::thread idle_server_thread([&] {
    idle_server_handshake = idle_server->Handshake(rank0, &server_error);
  });
  std::thread idle_client_thread([&] {
    idle_client_handshake = idle_client->Handshake(rank1, &client_error);
  });
  idle_server_thread.join();
  idle_client_thread.join();
  Require(idle_server_handshake && idle_client_handshake,
          "TP idle pair handshakes");

  bool idle_command_received = false;
  TpControlCommand idle_command;
  std::string idle_error;
  std::thread idle_worker([&] {
    idle_command_received =
        idle_client->ReceiveCommand(&idle_command, &idle_error);
  });
  std::this_thread::sleep_for(kShortIoTimeout * 4);
  Require(idle_server->SendCommand(command, &server_error), server_error);
  idle_worker.join();
  Require(idle_command_received && idle_command.sequence == command.sequence,
          "TP worker survives an idle wait longer than its I/O timeout: " +
              idle_error);

  TpResponseBroker idle_broker(idle_server, 1);
  Require(idle_broker.RegisterPendingResponse(command.sequence, &server_error),
          server_error);
  std::this_thread::sleep_for(kShortIoTimeout * 4);
  const TpControlResponse idle_response{.sequence = command.sequence,
                                        .tokens = {20, 21}};
  Require(idle_client->SendResponse(idle_response, &client_error),
          client_error);
  TpControlResponse idle_received;
  Require(idle_broker.WaitForResponse(command.sequence, &idle_received,
                                      &server_error),
          "TP broker survives an idle wait longer than its I/O timeout: " +
              server_error);
  Require(idle_received.tokens == idle_response.tokens,
          "TP broker delivers the response that arrived after the idle wait");

  Require(server->port() == port && client->port() == port,
          "TP control service port");

  std::puts(
      "PASS: TP control handshake, C1/C2 envelopes, instructions, and "
      "broker routing");
  return 0;
}

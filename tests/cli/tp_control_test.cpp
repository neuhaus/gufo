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
using gufo::server::TpControlStepConsumer;
using gufo::server::TpControlStepPublisher;
using gufo::server::TpControlCommand;
using gufo::server::TpControlCommandKind;
using gufo::server::TpControlConfig;
using gufo::server::TpControlResponse;
using gufo::server::TpControlResponseKind;
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

  TpControlConfig rank0{.rank = 0,
                        .world_size = 2,
                        .max_context = 4096,
                        .max_draft_tokens = 7,
                        .use_mtp = true,
                        .allow_cache_reuse = true,
                        .auth_token = "test-token",
                        .prefill_chunk_tokens = 512};
  TpControlConfig rank1 = rank0;
  rank1.rank = 1;
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

  // kStep is the per-token message that makes rank 0 authoritative: it carries
  // the token rank 0 sampled so rank 1 feeds that instead of sampling its own.
  // It belongs to the request named by `sequence` and to no plan of its own, so
  // the round trip must preserve the three step fields and invent no members.
  const TpControlCommand step{.sequence = command.sequence,
                              .kind = TpControlCommandKind::kStep,
                              .step_token = 4242,
                              .step_index = 9,
                              .step_final = true};
  std::string step_error;
  Require(ValidateTpControlCommand(step, &step_error), step_error);
  Require(server->SendCommand(step, &server_error), server_error);
  TpControlCommand received_step;
  Require(client->ReceiveCommand(&received_step, &client_error), client_error);
  Require(received_step.sequence == step.sequence &&
              received_step.kind == TpControlCommandKind::kStep &&
              received_step.members.empty() &&
              received_step.step_token == step.step_token &&
              received_step.step_index == step.step_index &&
              received_step.step_final == step.step_final,
          "TP step command round trip");

  // A non-final step must not be mistaken for a final one, and a negative
  // token must survive the u32 round trip by bit pattern rather than clamping.
  const TpControlCommand mid_step{.sequence = command.sequence,
                                  .kind = TpControlCommandKind::kStep,
                                  .step_token = -7,
                                  .step_index = 0,
                                  .step_final = false};
  Require(server->SendCommand(mid_step, &server_error), server_error);
  TpControlCommand received_mid;
  Require(client->ReceiveCommand(&received_mid, &client_error), client_error);
  Require(received_mid.step_token == -7 && received_mid.step_index == 0 &&
              !received_mid.step_final,
          "TP step command preserves a negative token and a non-final flag");

  // Every field below belongs to the request a step belongs to, not to the step
  // itself. A step carrying one is a disagreement between the two messages, so
  // it must be refused rather than reconciled -- and refused on the wire, not
  // only by the validator, because that is the path a peer exercises.
  TpPlanDigest nonzero_digest{};
  nonzero_digest[0] = 1;
  const auto refuses_step = [&](const std::string& what,
                                const TpControlCommand& bad) {
    std::string why;
    Require(!ValidateTpControlCommand(bad, &why) && !why.empty(),
            "TP step validation must refuse " + what + ", but said: " + why);
    std::string send_why;
    Require(!server->SendCommand(bad, &send_why) && !send_why.empty(),
            "TP step send must refuse " + what + ", but said: " + send_why);
  };
  {
    TpControlCommand bad = step;
    bad.sequence = 0;
    refuses_step("a zero sequence", bad);
  }
  {
    TpControlCommand bad = step;
    bad.members = {{.member_id = command.sequence, .max_tokens = 4}};
    refuses_step("a member", bad);
  }
  {
    TpControlCommand bad = step;
    bad.prompt_tokens = {10, 11, 12};
    refuses_step("a prompt", bad);
  }
  {
    TpControlCommand bad = step;
    bad.max_tokens = 4;
    refuses_step("a token budget", bad);
  }
  {
    TpControlCommand bad = step;
    bad.cache_prompt = true;
    refuses_step("a cache request", bad);
  }
  {
    TpControlCommand bad = step;
    bad.cache_prefix_tokens = 2;
    refuses_step("a cache prefix", bad);
  }
  {
    TpControlCommand bad = step;
    bad.client_id = "probe";
    refuses_step("a client id", bad);
  }
  {
    TpControlCommand bad = step;
    bad.cohort_id = 3;
    refuses_step("a cohort scope", bad);
  }
  {
    TpControlCommand bad = step;
    bad.execution_plan_digest = nonzero_digest;
    refuses_step("an execution-plan digest", bad);
  }
  {
    TpControlCommand bad = step;
    bad.cache_plan_digest = nonzero_digest;
    refuses_step("a cache-plan digest", bad);
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

  // A bounded receive must return a command that arrives inside the bound, and
  // must hand the socket back to the unbounded command wait afterwards rather
  // than leaking a per-token bound into the next one.
  {
    const TpControlCommand prompt_step{
        .sequence = command.sequence,
        .kind = TpControlCommandKind::kStep,
        .step_token = 1234,
        .step_index = 3,
    };
    std::thread prompt_sender(
        [&] { (void)server->SendCommand(prompt_step, &server_error); });
    TpControlCommand within;
    std::string within_error;
    Require(client->ReceiveCommandWithin(&within, std::chrono::seconds(10),
                                         &within_error),
            "TP bounded receive must return a command inside its bound: " +
                within_error);
    prompt_sender.join();
    Require(within.kind == TpControlCommandKind::kStep &&
                within.step_token == 1234 && within.step_index == 3,
            "TP bounded receive preserves the step it waited for");

    // The next command is not sent yet, so this must simply keep waiting: the
    // bound applied to the step must not have become the command wait's bound.
    std::thread late_sender([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(400));
      (void)server->SendCommand(command, &server_error);
    });
    TpControlCommand late;
    std::string late_error;
    Require(client->ReceiveCommand(&late, &late_error),
            "TP command wait is unbounded again after a bounded receive: " +
                late_error);
    late_sender.join();
    Require(late.sequence == command.sequence,
            "TP unbounded command wait returns the command sent after the step");
  }

  // A bounded receive that expires must fail closed, and must leave the channel
  // unusable rather than letting a later read reinterpret the tail of a partial
  // frame. Nothing is sent here, so the bound is reached with an empty stream.
  {
    TpControlCommand expired;
    std::string expired_error;
    Require(!client->ReceiveCommandWithin(&expired,
                                          std::chrono::milliseconds(150),
                                          &expired_error) &&
                !expired_error.empty(),
            "TP bounded receive must fail when the bound expires");

    // Poisoned: a further receive fails immediately instead of blocking or
    // succeeding. If the stream were merely abandoned this would either hang
    // forever or consume a later frame as if it were whole.
    std::thread late_sender(
        [&] { (void)server->SendCommand(command, &server_error); });
    TpControlCommand after_poison;
    std::string poison_error;
    const bool poisoned = client->ReceiveCommandWithin(
        &after_poison, std::chrono::milliseconds(150), &poison_error);
    late_sender.join();
    Require(!poisoned && !poison_error.empty(),
            "TP bounded receive must stay refused after a timeout: " +
                poison_error);
  }

  // The step bridge needs its own channel pair: the bounded-receive test above
  // deliberately poisoned the first client, and a poisoned channel refuses
  // every later receive by design.
  {
    const auto bridge_port = FreePort();
    std::string bridge_client_error;
    std::shared_ptr<TpControlChannel> bridge_client;
    std::thread bridge_connector([&] {
      bridge_client =
          TpControlChannel::Connect("127.0.0.1", bridge_port, &bridge_client_error);
    });
    std::string bridge_server_error;
    auto bridge_server =
        TpControlChannel::Listen(bridge_port, &bridge_server_error);
    bridge_connector.join();
    Require(bridge_server != nullptr && bridge_client != nullptr,
            "TP step bridge pair connects");
    bool bridge_server_handshake = false;
    bool bridge_client_handshake = false;
    std::thread bridge_server_thread([&] {
      bridge_server_handshake = bridge_server->Handshake(rank0, &bridge_server_error);
    });
    std::thread bridge_client_thread([&] {
      bridge_client_handshake = bridge_client->Handshake(rank1, &bridge_client_error);
    });
    bridge_server_thread.join();
    bridge_client_thread.join();
    Require(bridge_server_handshake && bridge_client_handshake,
            "TP step bridge pair handshakes: " + bridge_server_error);

    TpControlStepPublisher publisher(bridge_server);
    TpControlStepConsumer consumer(bridge_client);
    const std::uint64_t exchange = 42;

    // A token published by rank 0 is the token rank 1 decodes, in order.
    for (const std::int32_t token : {11, 22, 33}) {
      std::string publish_error;
      std::thread publish_thread(
          [&] { (void)publisher.Publish(exchange, token, false, &publish_error); });
      std::int32_t consumed = 0;
      bool final = true;
      std::string consume_error;
      Require(consumer.Consume(exchange, &consumed, &final,
                               std::chrono::seconds(10), &consume_error),
              "TP step bridge delivers a published token: " + consume_error);
      publish_thread.join();
      Require(consumed == token && !final,
              "TP step bridge returns rank 0's token, not its own");
    }

    // A step naming another request must be refused: the protocol proves it is
    // well formed, but only the in-flight sequence proves it belongs here.
    {
      std::string publish_error;
      std::thread publish_thread([&] {
        (void)publisher.Publish(exchange + 1, 44, false, &publish_error);
      });
      std::int32_t consumed = 0;
      bool final = false;
      std::string consume_error;
      Require(!consumer.Consume(exchange, &consumed, &final,
                                std::chrono::seconds(10), &consume_error) &&
                  !consume_error.empty(),
              "TP step bridge must refuse a step for another request");
      publish_thread.join();
    }

    // Rank 0 ending the exchange is a failure for the consumer, not an empty
    // token: it has nothing to hand the scheduler and must not pretend
    // otherwise.
    {
      std::string publish_error;
      std::thread publish_thread([&] {
        (void)publisher.Publish(exchange, 0, true, &publish_error);
      });
      std::int32_t consumed = 0;
      bool final = false;
      std::string consume_error;
      Require(!consumer.Consume(exchange, &consumed, &final,
                                std::chrono::seconds(10), &consume_error) &&
                  !consume_error.empty(),
              "TP step bridge must fail when rank 0 ends the exchange");
      publish_thread.join();
    }
  }

  std::puts("PASS: TP control handshake, C1/C2 envelopes, step messages, "
            "bounded receive, step bridge, and broker routing");
  return 0;
}

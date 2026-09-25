#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "src/cli/serve/tp_control.hpp"

namespace {

using gufo::server::TpControlChannel;
using gufo::server::TpControlCommand;
using gufo::server::TpControlConfig;
using gufo::server::TpControlResponse;
using gufo::server::TpResponseBroker;

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
  std::thread server_handshake_thread([&] {
    server_handshake = server->Handshake(rank0, &server_error);
  });
  std::thread client_handshake_thread([&] {
    client_handshake = client->Handshake(rank1, &client_error);
  });
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
              received.max_tokens == command.max_tokens &&
              received.cache_prompt == command.cache_prompt &&
              received.cache_prefix_tokens == command.cache_prefix_tokens &&
              received.prompt_tokens == command.prompt_tokens &&
              received.client_id == command.client_id,
          "TP control command round trip");

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
              received_response.tokens == response.tokens &&
              received_response.draft_tokens == response.draft_tokens &&
              received_response.draft_accepted_tokens ==
                  response.draft_accepted_tokens &&
              received_response.cached_prompt_tokens ==
                  response.cached_prompt_tokens &&
              received_response.cache_snapshot_bytes ==
                  response.cache_snapshot_bytes,
          "TP control response round trip");

  const auto mismatch_port = FreePort();
  std::shared_ptr<TpControlChannel> mismatch_client;
  std::thread mismatch_connector([&] {
    mismatch_client = TpControlChannel::Connect("127.0.0.1", mismatch_port,
                                                 &client_error);
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
    broker_client = TpControlChannel::Connect("127.0.0.1", broker_port,
                                               &client_error);
  });
  auto broker_server = TpControlChannel::Listen(broker_port, &server_error);
  broker_connector.join();
  Require(broker_server != nullptr && broker_client != nullptr,
          "TP broker pair connects");
  bool broker_server_handshake = false;
  bool broker_client_handshake = false;
  std::thread broker_server_thread([&] {
    broker_server_handshake =
        broker_server->Handshake(rank0, &server_error);
  });
  std::thread broker_client_thread([&] {
    broker_client_handshake =
        broker_client->Handshake(rank1, &client_error);
  });
  broker_server_thread.join();
  broker_client_thread.join();
  Require(broker_server_handshake && broker_client_handshake,
          "TP broker pair handshakes");

  TpResponseBroker broker(broker_server, 3);
  Require(broker.RegisterPendingResponse(7, &server_error), server_error);
  Require(broker.RegisterPendingResponse(9, &server_error), server_error);
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
    command_received = broker_client->ReceiveCommand(&received_command,
                                                     &client_error);
  });
  const bool command_sent =
      broker_server->SendCommand(broker_command, &server_error);
  command_thread.join();
  Require(command_sent && command_received &&
              received_command.sequence == broker_command.sequence,
          "TP broker permits full-duplex command sending");

  bool responses_sent = false;
  std::thread response_thread([&] {
    const TpControlResponse nine{.sequence = 9, .tokens = {19}};
    const TpControlResponse seven{.sequence = 7, .tokens = {17}};
    const TpControlResponse eleven{.sequence = 11, .tokens = {11}};
    responses_sent = broker_client->SendResponse(nine, &client_error) &&
                     broker_client->SendResponse(seven, &client_error) &&
                     broker_client->SendResponse(eleven, &client_error);
  });
  TpControlResponse seven_response;
  TpControlResponse nine_response;
  TpControlResponse eleven_response;
  const bool seven_ready = broker.WaitForResponse(
      7, &seven_response, &server_error);
  const bool nine_ready = broker.WaitForResponse(
      9, &nine_response, &server_error);
  const bool eleven_ready = broker.WaitForResponse(
      11, &eleven_response, &server_error);
  response_thread.join();
  Require(responses_sent && seven_ready && nine_ready && eleven_ready &&
              seven_response.tokens == std::vector<std::int32_t>{17} &&
              nine_response.tokens == std::vector<std::int32_t>{19} &&
              eleven_response.tokens == std::vector<std::int32_t>{11},
          "TP broker routes out-of-order responses by sequence");

  Require(broker.RegisterPendingResponse(12, &server_error), server_error);
  const TpControlResponse unexpected{.sequence = 13, .tokens = {130}};
  Require(broker_client->SendResponse(unexpected, &client_error), client_error);
  TpControlResponse ignored;
  Require(!broker.WaitForResponse(12, &ignored, &server_error) &&
              server_error.find("unknown or duplicate") != std::string::npos,
          "TP broker fails closed on an unknown response");
  Require(!broker.RegisterPendingResponse(14, &server_error),
          "TP broker rejects registrations after poisoning");

  Require(server->port() == port && client->port() == port,
          "TP control service port");
  std::puts("PASS: TP control handshake and framed worker messages");
  return 0;
}

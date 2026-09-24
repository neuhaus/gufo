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
                        .auth_token = "test-token"};
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
                           .prompt_tokens = {10, 11, 12},
                           .client_id = "probe"};
  Require(server->SendCommand(command, &server_error), server_error);
  TpControlCommand received;
  Require(client->ReceiveCommand(&received, &client_error), client_error);
  Require(received.sequence == command.sequence &&
              received.max_tokens == command.max_tokens &&
              received.prompt_tokens == command.prompt_tokens &&
              received.client_id == command.client_id,
          "TP control command round trip");

  TpControlResponse response{.sequence = received.sequence,
                             .tokens = {20, 21},
                             .draft_tokens = 3,
                             .draft_accepted_tokens = 2,
                             .error = {}};
  Require(client->SendResponse(response, &client_error), client_error);
  TpControlResponse received_response;
  Require(server->ReceiveResponse(&received_response, &server_error),
          server_error);
  Require(received_response.sequence == response.sequence &&
              received_response.tokens == response.tokens &&
              received_response.draft_tokens == response.draft_tokens &&
              received_response.draft_accepted_tokens ==
                  response.draft_accepted_tokens,
          "TP control response round trip");

  Require(server->port() == port && client->port() == port,
          "TP control service port");
  std::puts("PASS: TP control handshake and framed worker messages");
  return 0;
}

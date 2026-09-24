#ifndef GUFO_SERVER_TP_CONTROL_HPP_
#define GUFO_SERVER_TP_CONTROL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace gufo::server {

struct TpControlConfig {
  std::uint32_t rank{0};
  std::uint32_t world_size{1};
  std::uint32_t max_context{4096};
  std::uint32_t max_draft_tokens{7};
  bool use_mtp{false};
  std::string auth_token;
};

struct TpControlCommand {
  std::uint64_t sequence{0};
  std::uint32_t max_tokens{0};
  std::vector<std::int32_t> prompt_tokens;
  std::string client_id;
};

struct TpControlResponse {
  std::uint64_t sequence{0};
  std::vector<std::int32_t> tokens;
  std::uint64_t draft_tokens{0};
  std::uint64_t draft_accepted_tokens{0};
  std::string error;
};

/// Ordered, versioned TCP control channel for the first TP=2 worker slice.
/// Tensor payload still uses the RDMA communicator; this channel carries only
/// prepared prompt commands, responses, and lifecycle handshakes.
class TpControlChannel final {
 public:
  [[nodiscard]] static std::shared_ptr<TpControlChannel> Listen(
      std::uint16_t port, std::string* error);
  [[nodiscard]] static std::shared_ptr<TpControlChannel> Connect(
      const std::string& host, std::uint16_t port, std::string* error);

  ~TpControlChannel();

  TpControlChannel(const TpControlChannel&) = delete;
  TpControlChannel& operator=(const TpControlChannel&) = delete;

  [[nodiscard]] bool Handshake(const TpControlConfig& config,
                               std::string* error);
  [[nodiscard]] bool SendCommand(const TpControlCommand& command,
                                 std::string* error);
  [[nodiscard]] bool ReceiveCommand(TpControlCommand* command,
                                    std::string* error);
  [[nodiscard]] bool SendResponse(const TpControlResponse& response,
                                  std::string* error);
  [[nodiscard]] bool ReceiveResponse(TpControlResponse* response,
                                     std::string* error);

  [[nodiscard]] std::uint16_t port() const noexcept;
  [[nodiscard]] std::uint32_t rank() const noexcept;
  [[nodiscard]] std::uint32_t world_size() const noexcept;

 private:
  TpControlChannel(int fd, std::uint32_t rank, std::uint16_t port);

  [[nodiscard]] bool SendFrame(std::uint16_t type, std::uint64_t sequence,
                               const std::vector<std::uint8_t>& payload,
                               std::string* error);
  [[nodiscard]] bool ReceiveFrame(std::uint16_t type, std::uint64_t* sequence,
                                  std::vector<std::uint8_t>* payload,
                                  std::string* error);
  [[nodiscard]] bool SendAll(const void* data, std::size_t bytes,
                             std::string* error);
  [[nodiscard]] bool RecvAll(void* data, std::size_t bytes, std::string* error);

  int fd_{-1};
  std::uint16_t port_{0};
  std::uint32_t rank_{0};
  std::uint32_t world_size_{1};
  std::uint32_t max_context_{0};
  bool handshaken_{false};
  std::string auth_token_;
  std::mutex io_mutex_;
  std::vector<std::uint8_t> command_prompt_;
  std::vector<std::uint8_t> response_payload_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_TP_CONTROL_HPP_

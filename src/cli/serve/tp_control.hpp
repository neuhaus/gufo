#ifndef GUFO_SERVER_TP_CONTROL_HPP_
#define GUFO_SERVER_TP_CONTROL_HPP_

#include <atomic>
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
  bool allow_cache_reuse{false};
  std::string auth_token;
  std::uint32_t prefill_chunk_tokens{512};
};

struct TpControlCommand {
  /// Response correlation key and C1 RDMA collective scope.
  std::uint64_t sequence{0};
  std::uint32_t max_tokens{0};
  bool cache_prompt{false};
  std::uint32_t cache_prefix_tokens{0};
  std::vector<std::int32_t> prompt_tokens;
  std::string client_id;
};

struct TpControlResponse {
  /// Correlates with the command and its bound C1 collective scope.
  std::uint64_t sequence{0};
  std::vector<std::int32_t> tokens;
  std::uint64_t draft_tokens{0};
  std::uint64_t draft_accepted_tokens{0};
  std::uint32_t cached_prompt_tokens{0};
  std::uint64_t cache_snapshot_bytes{0};
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
  /// Wake a blocked directional reader/writer during broker shutdown.
  void Interrupt() noexcept;

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
  std::mutex send_mutex_;
  std::mutex receive_mutex_;
  std::atomic<bool> interrupted_{false};
  std::vector<std::uint8_t> command_prompt_;
  std::vector<std::uint8_t> response_payload_;
};

/// Rank-zero response owner. The dedicated reader routes final responses by
/// control sequence; production construction uses capacity one. This is a
/// safety foundation for ordered C2 cohorts, not an enablement of C2 itself.
class TpResponseBroker final {
 public:
  TpResponseBroker(std::shared_ptr<TpControlChannel> control,
                   std::size_t max_pending_responses);
  ~TpResponseBroker();

  TpResponseBroker(const TpResponseBroker&) = delete;
  TpResponseBroker& operator=(const TpResponseBroker&) = delete;
  TpResponseBroker(TpResponseBroker&&) = delete;
  TpResponseBroker& operator=(TpResponseBroker&&) = delete;

  [[nodiscard]] bool RegisterPendingResponse(std::uint64_t sequence,
                                             std::string* error);
  [[nodiscard]] bool CancelUnsentResponse(std::uint64_t sequence,
                                          std::string* error);
  [[nodiscard]] bool WaitForResponse(std::uint64_t sequence,
                                     TpControlResponse* response,
                                     std::string* error);
  void FailAll(std::string reason);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_TP_CONTROL_HPP_

#include "src/models/qwen38_flash_next/kernels/rocm/verbs.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include <infiniband/verbs.h>

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

constexpr std::uint32_t kWireMagic = 0x47554654U;  // "GUFT"
constexpr std::uint32_t kWireVersion = 1;
constexpr std::size_t kBufferBytes = 64U << 20;
constexpr std::uint32_t kPort = 1;
constexpr auto kCollectiveTimeout = std::chrono::seconds(30);

void SetError(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = std::move(message);
  }
}

std::string SystemError(const char* operation) {
  return std::string(operation) + ": " + std::strerror(errno);
}

class Socket {
public:
  Socket() = default;
  explicit Socket(int fd) : fd_(fd) {}
  Socket(const Socket&) = delete;
  Socket& operator=(const Socket&) = delete;
  Socket(Socket&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  Socket& operator=(Socket&& other) noexcept {
    if (this != &other) {
      Close();
      fd_ = std::exchange(other.fd_, -1);
    }
    return *this;
  }
  ~Socket() { Close(); }

  [[nodiscard]] bool valid() const noexcept { return fd_ >= 0; }

  void Close() noexcept {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  [[nodiscard]] static Socket Listen(const std::string& host,
                                    std::uint16_t port, std::string* error) {
    struct addrinfo hints {};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    const std::string service = std::to_string(port);
    struct addrinfo* addresses = nullptr;
    const int result = ::getaddrinfo(host.empty() ? nullptr : host.c_str(),
                                     service.c_str(), &hints, &addresses);
    if (result != 0) {
      SetError(error, "bootstrap getaddrinfo: " +
                          std::string(::gai_strerror(result)));
      return {};
    }
    int fd = -1;
    for (auto* address = addresses; address != nullptr;
         address = address->ai_next) {
      fd = ::socket(address->ai_family, address->ai_socktype,
                    address->ai_protocol);
      if (fd < 0) {
        continue;
      }
      int reuse = 1;
      (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
      if (::bind(fd, address->ai_addr, address->ai_addrlen) == 0 &&
          ::listen(fd, 1) == 0) {
        break;
      }
      ::close(fd);
      fd = -1;
    }
    ::freeaddrinfo(addresses);
    if (fd < 0) {
      SetError(error, SystemError("bootstrap listen"));
      return {};
    }
    SetTimeouts(fd);
    int peer = -1;
    do {
      peer = ::accept(fd, nullptr, nullptr);
    } while (peer < 0 && errno == EINTR);
    if (peer < 0) {
      SetError(error, SystemError("bootstrap accept"));
      ::close(fd);
      return {};
    }
    Socket listener(fd);
    SetTimeouts(peer);
    return Socket(peer);
  }

  [[nodiscard]] static Socket Connect(const std::string& host,
                                     std::uint16_t port, std::string* error) {
    if (host.empty()) {
      SetError(error, "rank one requires a bootstrap host");
      return {};
    }
    const auto deadline = std::chrono::steady_clock::now() + kCollectiveTimeout;
    for (;;) {
      struct addrinfo hints {};
      hints.ai_family = AF_INET;
      hints.ai_socktype = SOCK_STREAM;
      const std::string service = std::to_string(port);
      struct addrinfo* addresses = nullptr;
      const int result =
          ::getaddrinfo(host.c_str(), service.c_str(), &hints, &addresses);
      if (result != 0) {
        SetError(error, "bootstrap getaddrinfo: " +
                            std::string(::gai_strerror(result)));
        return {};
      }
      int fd = -1;
      for (auto* address = addresses; address != nullptr;
           address = address->ai_next) {
        fd = ::socket(address->ai_family, address->ai_socktype,
                      address->ai_protocol);
        if (fd < 0) {
          continue;
        }
        if (::connect(fd, address->ai_addr, address->ai_addrlen) == 0) {
          break;
        }
        ::close(fd);
        fd = -1;
      }
      ::freeaddrinfo(addresses);
      if (fd >= 0) {
        SetTimeouts(fd);
        return Socket(fd);
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        SetError(error, SystemError("bootstrap connect"));
        return {};
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  [[nodiscard]] bool SendAll(const void* data, std::size_t bytes,
                            std::string* error) const {
    const auto* cursor = static_cast<const std::uint8_t*>(data);
    std::size_t sent = 0;
    while (sent < bytes) {
      const ssize_t n = ::send(fd_, cursor + sent, bytes - sent, MSG_NOSIGNAL);
      if (n < 0 && errno == EINTR) {
        continue;
      }
      if (n <= 0) {
        SetError(error, SystemError("bootstrap send"));
        return false;
      }
      sent += static_cast<std::size_t>(n);
    }
    return true;
  }

  [[nodiscard]] bool RecvAll(void* data, std::size_t bytes,
                            std::string* error) const {
    auto* cursor = static_cast<std::uint8_t*>(data);
    std::size_t received = 0;
    while (received < bytes) {
      const ssize_t n = ::recv(fd_, cursor + received, bytes - received, 0);
      if (n < 0 && errno == EINTR) {
        continue;
      }
      if (n <= 0) {
        SetError(error, SystemError("bootstrap receive"));
        return false;
      }
      received += static_cast<std::size_t>(n);
    }
    return true;
  }

private:
  static void SetTimeouts(int fd) noexcept {
    struct timeval timeout {};
    timeout.tv_sec = 30;
    timeout.tv_usec = 0;
    (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  }

  int fd_{-1};
};

struct Wire {
  std::uint32_t magic{kWireMagic};
  std::uint32_t version{kWireVersion};
  std::uint32_t rank{0};
  std::uint32_t world_size{0};
  std::uint32_t qp_num{0};
  std::uint16_t lid{0};
  std::uint8_t mtu{0};
  std::uint8_t gid_index{0};
  std::uint16_t port_num{kPort};
  ibv_gid sgid{};
  std::uint64_t remote_mr_address{0};
  std::uint32_t rkey{0};
};

class Ibrverbs final : public Communicator {
public:
  explicit Ibrverbs(const IbrverbsConfig& config) : config_(config) {}
  ~Ibrverbs() override { Cleanup(); }

  Ibrverbs(const Ibrverbs&) = delete;
  Ibrverbs& operator=(const Ibrverbs&) = delete;

  [[nodiscard]] bool Initialize(std::string* error) {
    if (config_.world_size != 2 || config_.rank > 1 ||
        config_.bootstrap_port == 0 ||
        config_.device_index > static_cast<std::uint32_t>(
                                  std::numeric_limits<int>::max())) {
      SetError(error,
               "verbs communicator requires two ranks and a bootstrap port");
      return false;
    }
    if (hipSetDevice(static_cast<int>(config_.device_index)) != hipSuccess) {
      SetError(error, "hipSetDevice for verbs communicator failed");
      return false;
    }
    int device_count = 0;
    ibv_device** devices = ibv_get_device_list(&device_count);
    if (devices == nullptr || device_count <= config_.device_index) {
      if (devices != nullptr) {
        ibv_free_device_list(devices);
      }
      SetError(error, "no usable InfiniBand device");
      return false;
    }
    context_ = ibv_open_device(devices[config_.device_index]);
    ibv_free_device_list(devices);
    if (context_ == nullptr) {
      SetError(error, "ibv_open_device failed");
      return false;
    }
    pd_ = ibv_alloc_pd(context_);
    if (pd_ == nullptr) {
      SetError(error, "ibv_alloc_pd failed");
      return false;
    }
    cq_ = ibv_create_cq(context_, 16, nullptr, nullptr, 0);
    if (cq_ == nullptr) {
      SetError(error, "ibv_create_cq failed");
      return false;
    }

    ibv_port_attr port {};
    if (ibv_query_port(context_, kPort, &port) != 0 ||
        port.state != IBV_PORT_ACTIVE) {
      SetError(error, "InfiniBand port 1 is not active");
      return false;
    }
    if (ibv_query_gid(context_, kPort, config_.gid_index, &local_sgid_) != 0) {
      SetError(error, "ibv_query_gid failed");
      return false;
    }

    if (hipHostMalloc(&send_buffer_, kBufferBytes,
                      hipHostMallocMapped | hipHostMallocPortable) !=
            hipSuccess ||
        hipHostMalloc(&recv_buffer_, kBufferBytes,
                      hipHostMallocMapped | hipHostMallocPortable) !=
            hipSuccess) {
      SetError(error, "hipHostMalloc for verbs buffers failed");
      return false;
    }
    send_mr_ = ibv_reg_mr(pd_, send_buffer_, kBufferBytes,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    recv_mr_ = ibv_reg_mr(pd_, recv_buffer_, kBufferBytes,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (send_mr_ == nullptr || recv_mr_ == nullptr) {
      SetError(error, "ibv_reg_mr for verbs buffers failed");
      return false;
    }

    struct ibv_qp_init_attr init {};
    init.send_cq = cq_;
    init.recv_cq = cq_;
    init.cap.max_send_wr = 16;
    init.cap.max_recv_wr = 16;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;
    init.qp_type = IBV_QPT_RC;
    init.sq_sig_all = 0;
    qp_ = ibv_create_qp(pd_, &init);
    if (qp_ == nullptr) {
      SetError(error, "ibv_create_qp failed");
      return false;
    }

    struct ibv_qp_attr attr {};
    attr.qp_state = IBV_QP_INIT;
    attr.pkey_index = 0;
    attr.port_num = kPort;
    attr.qp_access_flags = IBV_QP_ACCESS_LOCAL_WRITE |
                           IBV_QP_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(qp_, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                     IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
      SetError(error, "ibv_modify_qp INIT failed");
      return false;
    }

    Wire local{};
    local.rank = config_.rank;
    local.world_size = config_.world_size;
    local.qp_num = qp_->qp_num;
    local.lid = port.lid;
    local.mtu = static_cast<std::uint8_t>(port.active_mtu);
    local.gid_index = static_cast<std::uint8_t>(config_.gid_index);
    local.sgid = local_sgid_;
    local.remote_mr_address = reinterpret_cast<std::uint64_t>(recv_buffer_);
    local.rkey = recv_mr_->rkey;

    Socket control = config_.rank == 0
                         ? Socket::Listen(config_.bootstrap_host,
                                          config_.bootstrap_port, error)
                         : Socket::Connect(config_.bootstrap_host,
                                           config_.bootstrap_port, error);
    if (!control.valid()) {
      return false;
    }
    if (!control.SendAll(&local, sizeof(local), error)) {
      return false;
    }
    Wire remote{};
    if (!control.RecvAll(&remote, sizeof(remote), error)) {
      return false;
    }
    if (remote.magic != kWireMagic || remote.version != kWireVersion ||
        remote.world_size != config_.world_size ||
        remote.rank != 1U - config_.rank || remote.qp_num == 0 ||
        remote.mtu < IBV_MTU_256 || remote.mtu > IBV_MTU_4096 ||
        remote.remote_mr_address == 0 || remote.rkey == 0) {
      SetError(error, "verbs bootstrap metadata is invalid");
      return false;
    }

    const auto mtu = static_cast<ibv_mtu>(
        std::min(static_cast<unsigned>(local.mtu),
                 static_cast<unsigned>(remote.mtu)));
    attr = {};
    attr.qp_state = IBV_QP_RTR;
    attr.path_mtu = mtu;
    attr.dest_qp_num = remote.qp_num;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.dlid = remote.lid;
    attr.ah_attr.sgid_index = local.gid_index;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = kPort;
    std::memcpy(&attr.ah_attr.grh.dgid, &remote.sgid, sizeof(remote.sgid));
    if (ibv_modify_qp(qp_, &attr, IBV_QP_STATE | IBV_QP_AV |
                                     IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                     IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                                     IBV_QP_MIN_RNR_TIMER) != 0) {
      SetError(error, "ibv_modify_qp RTR failed");
      return false;
    }
    attr = {};
    attr.qp_state = IBV_QP_RTS;
    attr.sq_psn = 0;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = 1;
    if (ibv_modify_qp(qp_, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN |
                                     IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                     IBV_QP_RNR_RETRY |
                                     IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
      SetError(error, "ibv_modify_qp RTS failed");
      return false;
    }
    remote_address_ = remote.remote_mr_address;
    remote_rkey_ = remote.rkey;
    return true;
  }

  [[nodiscard]] bool AllReduceSum(float* data, std::size_t bytes,
                                 hipStream_t stream,
                                 std::string* error) override {
    std::lock_guard lock(mutex_);
    if (qp_ == nullptr) {
      SetError(error, "verbs communicator is not initialized");
      return false;
    }
    if (bytes > kBufferBytes || bytes % sizeof(float) != 0 ||
        (bytes != 0 && data == nullptr)) {
      SetError(error, "all-reduce buffer is invalid");
      return false;
    }
    if (bytes == 0) {
      return true;
    }
    if (hipStreamSynchronize(stream) != hipSuccess ||
        hipMemcpy(send_buffer_, data, bytes, hipMemcpyDeviceToHost) !=
            hipSuccess) {
      SetError(error, "HIP device-to-host all-reduce staging failed");
      return false;
    }
    for (std::size_t offset = 0; offset < bytes; offset += kBufferBytes) {
      const std::size_t count = std::min(kBufferBytes, bytes - offset);
      auto* source = static_cast<std::uint8_t*>(send_buffer_) + offset;
      auto* destination = static_cast<std::uint8_t*>(recv_buffer_) + offset;
      ibv_sge sge {};
      sge.addr = reinterpret_cast<std::uint64_t>(source);
      sge.length = static_cast<unsigned>(count);
      sge.lkey = send_mr_->lkey;
      ibv_send_wr wr {};
      wr.wr_id = static_cast<std::uint64_t>(offset);
      wr.sg_list = &sge;
      wr.num_sge = 1;
      wr.send_flags = IBV_SEND_SIGNALED;
      wr.opcode = IBV_WR_RDMA_WRITE;
      wr.wr.rdma.remote_addr = remote_address_ + offset;
      wr.wr.rdma.rkey = remote_rkey_;
      if (ibv_post_send(qp_, &wr, nullptr) != nullptr) {
        SetError(error, "ibv_post_send failed");
        return false;
      }
      if (!PollCompletion(offset, error)) {
        return false;
      }
      auto* local = reinterpret_cast<float*>(source);
      const auto* peer = reinterpret_cast<const float*>(destination);
      const std::size_t values = count / sizeof(float);
      for (std::size_t i = 0; i < values; ++i) {
        local[i] += peer[i];
      }
    }
    if (hipMemcpy(data, send_buffer_, bytes, hipMemcpyHostToDevice) !=
            hipSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
      SetError(error, "HIP host-to-device all-reduce staging failed");
      return false;
    }
    return true;
  }

private:
  [[nodiscard]] bool PollCompletion(std::uint64_t expected,
                                    std::string* error) const {
    const auto deadline = std::chrono::steady_clock::now() + kCollectiveTimeout;
    ibv_wc completion {};
    while (std::chrono::steady_clock::now() < deadline) {
      const int result = ibv_poll_cq(cq_, 1, &completion);
      if (result < 0) {
        SetError(error, "ibv_poll_cq failed");
        return false;
      }
      if (result == 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        continue;
      }
      if (completion.status != IBV_WC_SUCCESS) {
        SetError(error, "verbs all-reduce write failed with status " +
                            std::to_string(completion.status));
        return false;
      }
      if (completion.wr_id != expected) {
        SetError(error, "verbs completion sequence mismatch");
        return false;
      }
      return true;
    }
    SetError(error, "verbs all-reduce completion timed out");
    return false;
  }

  void Cleanup() noexcept {
    if (qp_ != nullptr) {
      ibv_destroy_qp(qp_);
      qp_ = nullptr;
    }
    if (send_mr_ != nullptr) {
      (void)ibv_dereg_mr(send_mr_);
      send_mr_ = nullptr;
    }
    if (recv_mr_ != nullptr) {
      (void)ibv_dereg_mr(recv_mr_);
      recv_mr_ = nullptr;
    }
    if (send_buffer_ != nullptr) {
      (void)hipHostFree(send_buffer_);
      send_buffer_ = nullptr;
    }
    if (recv_buffer_ != nullptr) {
      (void)hipHostFree(recv_buffer_);
      recv_buffer_ = nullptr;
    }
    if (cq_ != nullptr) {
      ibv_destroy_cq(cq_);
      cq_ = nullptr;
    }
    if (pd_ != nullptr) {
      ibv_dealloc_pd(pd_);
      pd_ = nullptr;
    }
    if (context_ != nullptr) {
      ibv_close_device(context_);
      context_ = nullptr;
    }
  }

  IbrverbsConfig config_;
  ibv_context* context_{nullptr};
  ibv_pd* pd_{nullptr};
  ibv_cq* cq_{nullptr};
  ibv_qp* qp_{nullptr};
  ibv_mr* send_mr_{nullptr};
  ibv_mr* recv_mr_{nullptr};
  void* send_buffer_{nullptr};
  void* recv_buffer_{nullptr};
  ibv_gid local_sgid_{};
  std::uint64_t remote_address_{0};
  std::uint32_t remote_rkey_{0};
  mutable std::mutex mutex_;
};

}  // namespace

std::shared_ptr<Communicator> CreateIbrverbsCommunicator(
    const IbrverbsConfig& config, std::string* error_msg) {
  auto communicator = std::make_shared<Ibrverbs>(config);
  if (!communicator->Initialize(error_msg)) {
    return nullptr;
  }
  return communicator;
}

}  // namespace gufo::models::qwen38_flash_next::rocm

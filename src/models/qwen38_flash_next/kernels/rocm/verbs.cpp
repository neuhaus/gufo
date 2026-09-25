#include "src/models/qwen38_flash_next/kernels/rocm/verbs.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include <infiniband/verbs.h>

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

constexpr std::uint32_t kWireMagic = 0x47554654U;  // "GUFT"
constexpr std::uint32_t kWireVersion = 3;
constexpr std::uint32_t kCollectiveMagic = 0x47554348U;  // "GUCH"
constexpr std::uint32_t kCollectiveVersion = 3;
constexpr std::uint32_t kReadyMagic = 0x47555244U;  // "GURD"
constexpr std::size_t kBufferBytes = 64U << 20;
constexpr std::uintptr_t kSendAddress = 0x0000700000000000ULL;
constexpr std::uintptr_t kRecvAddress = kSendAddress + kBufferBytes;
constexpr std::uintptr_t kResultAddress = kRecvAddress + kBufferBytes;
constexpr std::uint32_t kPort = 1;
constexpr auto kCollectiveTimeout = std::chrono::seconds(30);
constexpr int kMemoryAccessFlags =
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

void SetError(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = std::move(message);
  }
}

std::string SystemError(const char* operation) {
  return std::string(operation) + ": " + std::strerror(errno);
}

bool MapFixedBuffer(std::uintptr_t address, void** buffer,
                    std::string* error) {
  void* mapped = ::mmap(reinterpret_cast<void*>(address), kBufferBytes,
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1,
                        0);
  if (mapped == MAP_FAILED) {
    SetError(error, "fixed RDMA buffer mmap failed: " +
                        std::string(std::strerror(errno)));
    return false;
  }
  if (reinterpret_cast<std::uintptr_t>(mapped) != address) {
    ::munmap(mapped, kBufferBytes);
    SetError(error, "fixed RDMA buffer address mismatch");
    return false;
  }
  if (hipHostRegister(mapped, kBufferBytes, hipHostRegisterMapped) !=
      hipSuccess) {
    ::munmap(mapped, kBufferBytes);
    SetError(error, "hipHostRegister for fixed RDMA buffer failed");
    return false;
  }
  *buffer = mapped;
  return true;
}

void UnmapFixedBuffer(void* buffer) {
  if (buffer == nullptr) {
    return;
  }
  (void)hipHostUnregister(buffer);
  (void)::munmap(buffer, kBufferBytes);
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
    pollfd poll_fd{.fd = fd, .events = POLLIN, .revents = 0};
    const int poll_result = ::poll(&poll_fd, 1, 30000);
    if (poll_result <= 0) {
      SetError(error, poll_result == 0 ? "bootstrap accept timed out"
                                       : SystemError("bootstrap poll"));
      ::close(fd);
      return {};
    }
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
        const int flags = ::fcntl(fd, F_GETFL, 0);
        if (flags < 0 ||
            ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
          ::close(fd);
          fd = -1;
          continue;
        }
        bool connected = ::connect(fd, address->ai_addr, address->ai_addrlen) ==
                         0;
        if (!connected && errno == EINPROGRESS) {
          pollfd poll_fd{.fd = fd, .events = POLLOUT, .revents = 0};
          const int poll_result = ::poll(&poll_fd, 1, 1000);
          int socket_error = 0;
          socklen_t error_size = sizeof(socket_error);
          connected = poll_result > 0 &&
                      ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error,
                                   &error_size) == 0 &&
                      socket_error == 0;
        }
        if (connected) {
          if (::fcntl(fd, F_SETFL, flags) != 0) {
            ::close(fd);
            fd = -1;
            continue;
          }
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
    const int no_delay = 1;
    (void)::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &no_delay,
                       sizeof(no_delay));
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
  std::uint8_t link_layer{0};
  std::uint16_t port_num{kPort};
  ibv_gid sgid{};
  std::uint64_t remote_mr_address{0};
  std::uint32_t rkey{0};
};

struct CollectiveHeader {
  std::uint32_t magic{kCollectiveMagic};
  std::uint32_t version{kCollectiveVersion};
  std::uint64_t sequence{0};
  std::uint64_t bytes{0};
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
        config_.gid_index > std::numeric_limits<std::uint8_t>::max() ||
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
        port.state != IBV_PORT_ACTIVE ||
        port.link_layer != IBV_LINK_LAYER_INFINIBAND) {
      SetError(error, "native InfiniBand port 1 is not active");
      return false;
    }
    if (ibv_query_gid(context_, kPort, config_.gid_index, &local_sgid_) != 0) {
      SetError(error, "ibv_query_gid failed");
      return false;
    }

    if (!MapFixedBuffer(kSendAddress, &send_buffer_, error)) {
      return false;
    }
    send_registered_ = true;
    if (!MapFixedBuffer(kRecvAddress, &recv_buffer_, error)) {
      return false;
    }
    recv_registered_ = true;
    if (!MapFixedBuffer(kResultAddress, &result_buffer_, error)) {
      return false;
    }
    result_registered_ = true;
    send_mr_ = ibv_reg_mr(pd_, send_buffer_, kBufferBytes,
                          kMemoryAccessFlags);
    recv_mr_ = ibv_reg_mr(pd_, recv_buffer_, kBufferBytes,
                          kMemoryAccessFlags);
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
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = kPort;
    attr.qp_access_flags = kMemoryAccessFlags;
    if (ibv_modify_qp(qp_, &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS) != 0) {
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
    local.link_layer = static_cast<std::uint8_t>(port.link_layer);
    local.sgid = local_sgid_;
    // The peer reads the staged send window after the collective header says
    // that this rank's device-to-host copy is complete.
    local.remote_mr_address = reinterpret_cast<std::uint64_t>(send_buffer_);
    local.rkey = send_mr_->rkey;

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
        remote.port_num != kPort ||
        remote.link_layer != IBV_LINK_LAYER_INFINIBAND ||
        remote.mtu < IBV_MTU_256 || remote.mtu > IBV_MTU_4096 ||
        remote.remote_mr_address != kSendAddress || remote.rkey == 0) {
      SetError(error, "verbs bootstrap metadata is invalid");
      return false;
    }

    const auto mtu = static_cast<ibv_mtu>(
        std::min(static_cast<unsigned>(local.mtu),
                 static_cast<unsigned>(remote.mtu)));
    attr = {};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = mtu;
    attr.dest_qp_num = remote.qp_num;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.dlid = remote.lid;
    attr.ah_attr.grh.hop_limit = 1;
    attr.ah_attr.grh.sgid_index = local.gid_index;
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
    attr.qp_state = IBV_QPS_RTS;
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
    remote_send_address_ = remote.remote_mr_address;
    remote_rkey_ = remote.rkey;
    const std::uint32_t ready = kReadyMagic;
    std::uint32_t peer_ready = 0;
    if (!control.SendAll(&ready, sizeof(ready), error) ||
        !control.RecvAll(&peer_ready, sizeof(peer_ready), error) ||
        peer_ready != kReadyMagic) {
      SetError(error, "verbs post-RTS readiness exchange failed");
      return false;
    }
    control_ = std::make_unique<Socket>(std::move(control));
    return true;
  }

  [[nodiscard]] std::uint32_t rank() const noexcept override {
    return config_.rank;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept override {
    return config_.world_size;
  }
  [[nodiscard]] int device_index() const noexcept override {
    return static_cast<int>(config_.device_index);
  }

  [[nodiscard]] bool AllReduceSum(float* data, std::size_t bytes,
                                 hipStream_t stream,
                                 std::string* error) override {
    std::lock_guard lock(mutex_);
    if (qp_ == nullptr) {
      SetError(error, "verbs communicator is not initialized");
      return false;
    }
    if (control_ == nullptr) {
      SetError(error, "verbs control channel is not initialized");
      return false;
    }
    if (bytes > kBufferBytes || bytes % sizeof(float) != 0 ||
        (bytes != 0 && data == nullptr)) {
      SetError(error, "all-reduce buffer is invalid");
      return false;
    }
    const std::uint64_t sequence = sequence_++;
    const CollectiveHeader outgoing{
        .sequence = sequence,
        .bytes = bytes,
    };
    CollectiveHeader incoming{};

    // Stage before announcing readiness. The peer can then read directly from
    // this send window without a per-chunk TCP data-ready round trip.
    if (bytes != 0 &&
        (hipStreamSynchronize(stream) != hipSuccess ||
         hipMemcpy(send_buffer_, data, bytes, hipMemcpyDeviceToHost) !=
             hipSuccess)) {
      SetError(error, "HIP device-to-host all-reduce staging failed");
      return false;
    }
    if (!control_->SendAll(&outgoing, sizeof(outgoing), error) ||
        !control_->RecvAll(&incoming, sizeof(incoming), error)) {
      return false;
    }
    if (incoming.magic != kCollectiveMagic ||
        incoming.version != kCollectiveVersion ||
        incoming.sequence != outgoing.sequence || incoming.bytes != bytes) {
      SetError(error, "verbs collective sequence or size mismatch");
      return false;
    }
    if (bytes == 0) {
      const std::uint32_t ack = kReadyMagic;
      std::uint32_t peer_ack = 0;
      return control_->SendAll(&ack, sizeof(ack), error) &&
             control_->RecvAll(&peer_ack, sizeof(peer_ack), error) &&
             peer_ack == kReadyMagic;
    }
    for (std::size_t offset = 0; offset < bytes; offset += kBufferBytes) {
      const std::size_t count = std::min(kBufferBytes, bytes - offset);
      auto* destination = static_cast<std::uint8_t*>(recv_buffer_) + offset;
      ibv_sge sge {};
      sge.addr = reinterpret_cast<std::uint64_t>(destination);
      sge.length = static_cast<unsigned>(count);
      sge.lkey = recv_mr_->lkey;
      ibv_send_wr wr {};
      wr.wr_id = static_cast<std::uint64_t>(offset);
      wr.sg_list = &sge;
      wr.num_sge = 1;
      wr.send_flags = IBV_SEND_SIGNALED;
      wr.opcode = IBV_WR_RDMA_READ;
      wr.wr.rdma.remote_addr = remote_send_address_ + offset;
      wr.wr.rdma.rkey = remote_rkey_;
      if (ibv_post_send(qp_, &wr, nullptr) != 0) {
        SetError(error, "ibv_post_send failed");
        return false;
      }
      if (!PollCompletion(offset, error)) {
        return false;
      }
      auto* local = reinterpret_cast<float*>(
          static_cast<std::uint8_t*>(send_buffer_) + offset);
      const auto* peer = reinterpret_cast<const float*>(destination);
      auto* result = reinterpret_cast<float*>(
          static_cast<std::uint8_t*>(result_buffer_) + offset);
      const std::size_t values = count / sizeof(float);
      for (std::size_t i = 0; i < values; ++i) {
        result[i] = local[i] + peer[i];
      }
    }
    // Receiving the peer acknowledgement means that its read of this send
    // window is complete. The result window is independent, so the local
    // host-to-device copy below can proceed without racing the peer.
    const std::uint32_t ack = kReadyMagic;
    std::uint32_t peer_ack = 0;
    if (!control_->SendAll(&ack, sizeof(ack), error) ||
        !control_->RecvAll(&peer_ack, sizeof(peer_ack), error) ||
        peer_ack != kReadyMagic) {
      if (error != nullptr && error->empty()) {
        *error = "verbs collective acknowledgement failed";
      }
      return false;
    }
    if (hipMemcpy(data, result_buffer_, bytes, hipMemcpyHostToDevice) !=
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
        SetError(error, "verbs all-reduce read failed with status " +
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
    control_.reset();
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
    if (send_registered_) {
      UnmapFixedBuffer(send_buffer_);
      send_registered_ = false;
    }
    send_buffer_ = nullptr;
    if (recv_registered_) {
      UnmapFixedBuffer(recv_buffer_);
      recv_registered_ = false;
    }
    recv_buffer_ = nullptr;
    if (result_registered_) {
      UnmapFixedBuffer(result_buffer_);
      result_registered_ = false;
    }
    result_buffer_ = nullptr;
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
  void* result_buffer_{nullptr};
  bool send_registered_{false};
  bool recv_registered_{false};
  bool result_registered_{false};
  ibv_gid local_sgid_{};
  std::uint64_t remote_send_address_{0};
  std::uint32_t remote_rkey_{0};
  std::unique_ptr<Socket> control_;
  std::uint64_t sequence_{0};
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

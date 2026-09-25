#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_VERBS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_VERBS_HPP_

#include <cstdint>
#include <memory>
#include <string>

#include "src/models/qwen38_flash_next/kernels/rocm/communicator.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

struct IbrverbsConfig {
  std::uint32_t rank{0};
  std::uint32_t world_size{2};
  /// TCP bootstrap address used by rank 1. Rank 0 binds to all interfaces.
  std::string bootstrap_host;
  std::uint16_t bootstrap_port{18515};
  std::uint32_t device_index{0};
  std::uint32_t gid_index{0};
};

/// Creates a two-rank synchronous RC communicator. Tensor data uses RDMA
/// reads and an event-driven completion channel when available; the
/// bootstrap endpoint remains as a small ordered control/ack channel. Both
/// processes map identical fixed send, receive, and result host IOVA
/// windows.
[[nodiscard]] std::shared_ptr<Communicator> CreateIbrverbsCommunicator(
    const IbrverbsConfig& config, std::string* error_msg);

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_VERBS_HPP_

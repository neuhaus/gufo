#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_COMMUNICATOR_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_COMMUNICATOR_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <string>

namespace gufo::models::qwen38_flash_next::rocm {

/// Ordered communication boundary used by the model executor. The callback
/// receives a GPU-visible float buffer and must enqueue or complete the sum
/// in the supplied HIP stream before returning success.
class Communicator {
public:
  virtual ~Communicator() = default;
  virtual bool AllReduceSum(float* data, std::size_t bytes, hipStream_t stream,
                            std::string* error) = 0;
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_COMMUNICATOR_HPP_

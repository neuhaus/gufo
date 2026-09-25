#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_COMMUNICATOR_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_COMMUNICATOR_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace gufo::models::qwen38_flash_next::rocm {

/// Ordered communication boundary used by the model executor. The callback
/// receives a GPU-visible float buffer and must enqueue or complete the sum
/// in the supplied HIP stream before returning success.
///
/// A distributed communicator requires one exclusive operation scope for the
/// lifetime of a control request. The scope is an opaque logical identity; the
/// current TP2 serving path uses the control command sequence as that value.
/// The communicator adds and validates an operation ordinal for every
/// AllReduceSum call, so a collective cannot be silently associated with a
/// different request.
class Communicator {
public:
  virtual ~Communicator() = default;
  [[nodiscard]] virtual std::uint32_t rank() const noexcept { return 0; }
  [[nodiscard]] virtual std::uint32_t world_size() const noexcept { return 1; }
  [[nodiscard]] virtual int device_index() const noexcept { return 0; }
  [[nodiscard]] virtual bool BeginOperation(std::uint64_t scope_id,
                                            std::string* error) = 0;
  [[nodiscard]] virtual bool EndOperation(std::uint64_t scope_id,
                                          std::string* error) = 0;
  virtual bool AllReduceSum(float* data, std::size_t bytes, hipStream_t stream,
                            std::string* error) = 0;
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_COMMUNICATOR_HPP_

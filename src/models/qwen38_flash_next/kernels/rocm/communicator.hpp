#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_COMMUNICATOR_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_COMMUNICATOR_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace gufo::models::qwen38_flash_next::rocm {

/// Ordered communication boundary used by the model executor. A two-rank sum
/// is one `ExchangePartial` followed by adding the returned peer partial to
/// this rank's partial on the GPU. Each rank adds local plus peer, and float
/// addition of two operands is commutative, so both ranks hold identical sums.
///
/// A distributed communicator requires one exclusive operation scope for the
/// lifetime of a control request. The scope is an opaque logical identity; the
/// current TP2 serving path uses the control command sequence as that value.
/// The communicator adds and validates an operation ordinal for every
/// exchange, so a collective cannot be silently associated with a different
/// request.
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
  /// Sends this rank's `bytes` of `data`, complete once `stream` drains, and
  /// returns a GPU-readable pointer to the peer's partial of the same size, or
  /// null on failure. The pointer stays valid until the next exchange, which
  /// synchronizes `stream` before it reuses the buffer, so every exchange must
  /// use the same stream and the caller's add must be queued on it.
  [[nodiscard]] virtual const float* ExchangePartial(const float* data,
                                                     std::size_t bytes,
                                                     hipStream_t stream,
                                                     std::string* error) = 0;
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_COMMUNICATOR_HPP_

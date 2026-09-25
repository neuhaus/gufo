#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_DISTRIBUTED_TP_PARTITION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_DISTRIBUTED_TP_PARTITION_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next::distributed {

/// Equal contiguous expert ownership for one rank. The global expert count
/// remains the routing count; expert_begin/expert_count describe the local
/// encoded tensor range.
struct TpPartition {
  std::uint32_t rank{0};
  std::uint32_t world_size{1};
  std::uint32_t num_experts{0};
  std::uint32_t expert_begin{0};
  std::uint32_t expert_count{0};

  [[nodiscard]] bool distributed() const noexcept { return world_size > 1; }

  [[nodiscard]] bool Valid() const noexcept {
    return num_experts != 0 && world_size != 0 && rank < world_size &&
           num_experts % world_size == 0 &&
           expert_count == num_experts / world_size &&
           expert_begin == expert_count * rank;
  }

  /// Returns a local expert ID, or -1 when the global expert is owned by the
  /// peer rank. The value is intentionally suitable for the existing MMQ
  /// inactive-expert convention.
  [[nodiscard]] std::int32_t LocalExpert(std::uint32_t global) const noexcept {
    if (global < expert_begin || global - expert_begin >= expert_count) {
      return -1;
    }
    return static_cast<std::int32_t>(global - expert_begin);
  }

  /// Builds an equal contiguous partition. Uneven expert counts are rejected
  /// rather than silently assigning a different geometry to each rank.
  [[nodiscard]] static std::optional<TpPartition> Create(
      std::uint32_t num_experts, std::uint32_t rank, std::uint32_t world_size,
      std::string* error_msg);
};

/// A contiguous encoded range relative to the beginning of a TensorRef. The
/// range is suitable for WeightUpload::Copy after adding file_offset.
struct TensorRange {
  std::uint32_t first_expert{0};
  std::uint32_t expert_count{0};
  std::uint64_t byte_offset{0};
  std::size_t byte_size{0};
};

struct RoutedWeightPlan {
  std::size_t full_encoded_bytes;
  std::size_t local_encoded_bytes;
};

/// Computes the exact encoded byte count of the routed tensors and the local
/// expert subset selected by `partition`. This is a source-artifact estimate;
/// DeviceModel::resident_bytes() remains the authoritative post-conversion
/// device-memory measurement.
[[nodiscard]] std::optional<RoutedWeightPlan> PlanRoutedBytes(
    const ModelWeights& weights, const TpPartition& partition,
    const MtpWeights* mtp_weights = nullptr, std::string* error = nullptr);

/// Selects the local expert range without decoding or repacking the GGUF
/// tensor. RowBytes() supplies the format-specific geometry (Q8_0, Q4_K,
/// F32, and the other formats accepted by the model binder).
[[nodiscard]] std::optional<TensorRange> LocalExpertRange(
    const TensorRef& tensor, const TpPartition& partition,
    std::string* error_msg);

}  // namespace gufo::models::qwen38_flash_next::distributed

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_DISTRIBUTED_TP_PARTITION_HPP_

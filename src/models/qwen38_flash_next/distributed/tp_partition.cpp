#include "src/models/qwen38_flash_next/distributed/tp_partition.hpp"

#include <limits>
#include <utility>

namespace gufo::models::qwen38_flash_next::distributed {
namespace {

void SetError(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = std::move(message);
  }
}

}  // namespace

std::optional<TpPartition> TpPartition::Create(std::uint32_t num_experts,
                                               std::uint32_t rank,
                                               std::uint32_t world_size,
                                               std::string* error_msg) {
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  if (num_experts == 0 || world_size == 0 || rank >= world_size ||
      num_experts % world_size != 0) {
    SetError(error_msg,
             "TP expert partition requires a nonzero, evenly divisible expert "
             "count and a valid rank");
    return std::nullopt;
  }
  TpPartition result;
  result.rank = rank;
  result.world_size = world_size;
  result.num_experts = num_experts;
  result.expert_count = num_experts / world_size;
  result.expert_begin = result.expert_count * rank;
  return result;
}

std::optional<TensorRange> LocalExpertRange(const TensorRef& tensor,
                                            const TpPartition& partition,
                                            std::string* error_msg) {
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  if (!partition.Valid() || tensor.empty() ||
      tensor.experts != partition.num_experts) {
    SetError(error_msg,
             "TP expert range tensor does not match the partition geometry");
    return std::nullopt;
  }
  const std::size_t row_bytes = tensor.RowBytes();
  if (row_bytes == 0) {
    SetError(error_msg,
             "TP expert range has an unsupported tensor format or shape");
    return std::nullopt;
  }
  const std::uint64_t rows = tensor.rows;
  if (rows != 0 && partition.expert_count >
                        std::numeric_limits<std::uint64_t>::max() / rows) {
    SetError(error_msg, "TP expert range row geometry overflows");
    return std::nullopt;
  }
  const std::uint64_t local_rows = rows * partition.expert_count;
  if (local_rows != 0 &&
      static_cast<std::uint64_t>(row_bytes) >
          std::numeric_limits<std::uint64_t>::max() / local_rows) {
    SetError(error_msg, "TP expert range byte geometry overflows");
    return std::nullopt;
  }
  if (rows != 0 && partition.expert_begin >
                        std::numeric_limits<std::uint64_t>::max() / rows) {
    SetError(error_msg, "TP expert range offset geometry overflows");
    return std::nullopt;
  }
  const std::uint64_t offset_rows = rows * partition.expert_begin;
  if (offset_rows != 0 &&
      static_cast<std::uint64_t>(row_bytes) >
          std::numeric_limits<std::uint64_t>::max() / offset_rows) {
    SetError(error_msg, "TP expert range offset byte geometry overflows");
    return std::nullopt;
  }
  const std::uint64_t byte_offset = offset_rows * row_bytes;
  const std::uint64_t byte_size = local_rows * row_bytes;
  if (byte_offset > std::numeric_limits<std::size_t>::max() ||
      byte_size > std::numeric_limits<std::size_t>::max()) {
    SetError(error_msg, "TP expert range exceeds the host addressable size");
    return std::nullopt;
  }
  return TensorRange{partition.expert_begin, partition.expert_count,
                     byte_offset, static_cast<std::size_t>(byte_size)};
}

}  // namespace gufo::models::qwen38_flash_next::distributed

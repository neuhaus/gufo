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

bool CheckedAdd(std::size_t* total, std::size_t value, const char* label,
                std::string* error_msg) {
  if (*total > std::numeric_limits<std::size_t>::max() - value) {
    SetError(error_msg,
             std::string("TP routed weight plan overflows in ") + label);
    return false;
  }
  *total += value;
  return true;
}

std::optional<std::size_t> EncodedSize(const TensorRef& tensor,
                                       std::string* error_msg) {
  const std::size_t row_bytes = tensor.RowBytes();
  if (tensor.empty() || row_bytes == 0 || tensor.rows == 0 ||
      tensor.experts == 0) {
    SetError(error_msg, "TP routed weight plan has an invalid tensor shape");
    return std::nullopt;
  }
  if (tensor.rows > std::numeric_limits<std::size_t>::max() / row_bytes) {
    SetError(error_msg, "TP routed weight plan row bytes overflow");
    return std::nullopt;
  }
  const std::size_t rows = static_cast<std::size_t>(tensor.rows);
  const std::size_t row_total = rows * row_bytes;
  if (tensor.experts >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() /
                                 (row_total == 0 ? 1 : row_total))) {
    SetError(error_msg, "TP routed weight plan expert bytes overflow");
    return std::nullopt;
  }
  return row_total * static_cast<std::size_t>(tensor.experts);
}

bool AddRoutedTensor(const TensorRef& tensor, const TpPartition& partition,
                     RoutedWeightPlan* plan, std::string* error_msg) {
  const auto full = EncodedSize(tensor, error_msg);
  const auto range = LocalExpertRange(tensor, partition, error_msg);
  if (!full || !range) {
    return false;
  }
  return CheckedAdd(&plan->full_encoded_bytes, *full, "full tensor",
                    error_msg) &&
         CheckedAdd(&plan->local_encoded_bytes, range->byte_size,
                    "local tensor", error_msg);
}

bool AddLayerRouted(const LayerWeights& layer, const TpPartition& partition,
                    RoutedWeightPlan* plan, std::string* error_msg) {
  return AddRoutedTensor(layer.ffn_gate_exps, partition, plan, error_msg) &&
         AddRoutedTensor(layer.ffn_up_exps, partition, plan, error_msg) &&
         AddRoutedTensor(layer.ffn_down_exps, partition, plan, error_msg);
}

}  // namespace

std::optional<RoutedWeightPlan> PlanRoutedBytes(const ModelWeights& weights,
                                                const TpPartition& partition,
                                                const MtpWeights* mtp_weights,
                                                std::string* error_msg) {
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  if (!partition.Valid() ||
      partition.num_experts != weights.config.num_experts ||
      (mtp_weights != nullptr &&
       mtp_weights->config.num_experts != partition.num_experts)) {
    SetError(error_msg,
             "TP routed weight plan geometry does not match the model");
    return std::nullopt;
  }
  RoutedWeightPlan plan{};
  for (const auto& layer : weights.layers) {
    if (!AddLayerRouted(layer, partition, &plan, error_msg)) {
      return std::nullopt;
    }
  }
  if (mtp_weights != nullptr &&
      !AddLayerRouted(mtp_weights->block, partition, &plan, error_msg)) {
    return std::nullopt;
  }
  return plan;
}

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

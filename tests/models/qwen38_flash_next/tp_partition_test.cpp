#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>

#include "src/models/qwen38_flash_next/distributed/tp_partition.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::models::qwen38_flash_next::TensorRef;
using gufo::models::qwen38_flash_next::distributed::LocalExpertRange;
using gufo::models::qwen38_flash_next::distributed::TpPartition;

void Require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

void RequireError(const std::optional<TpPartition>& result,
                  const char* message) {
  Require(!result.has_value(), message);
}

}  // namespace

int main() {
  std::string error;
  const auto single = TpPartition::Create(512, 0, 1, &error);
  Require(single.has_value(), error.c_str());
  Require(!single->distributed(), "world size one is not distributed");
  Require(single->LocalExpert(0) == 0 && single->LocalExpert(511) == 0,
          "single-rank mapping");

  const auto first = TpPartition::Create(512, 0, 2, &error);
  const auto second = TpPartition::Create(512, 1, 2, &error);
  Require(first.has_value() && second.has_value(), "TP=2 partition");
  Require(first->expert_begin == 0 && first->expert_count == 256,
          "rank zero range");
  Require(second->expert_begin == 256 && second->expert_count == 256,
          "rank one range");
  Require(first->LocalExpert(0) == 0 && first->LocalExpert(255) == 255,
          "rank zero local IDs");
  Require(first->LocalExpert(256) == -1 && first->LocalExpert(511) == -1,
          "rank zero rejects peer IDs");
  Require(second->LocalExpert(0) == -1 && second->LocalExpert(256) == 0 &&
              second->LocalExpert(511) == 255,
          "rank one local IDs");

  const void* data = reinterpret_cast<const void*>(0x1000);
  const auto small_first = TpPartition::Create(4, 0, 2, &error);
  const auto small_second = TpPartition::Create(4, 1, 2, &error);
  Require(small_first.has_value() && small_second.has_value(),
          "small TP=2 partition");
  const TensorRef q4{data, GgmlType::kQ4_K, 256, 2, 4, 0, 0, "q4"};
  const auto q4_first = LocalExpertRange(q4, *small_first, &error);
  const auto q4_second = LocalExpertRange(q4, *small_second, &error);
  Require(q4_first.has_value() && q4_second.has_value(), error.c_str());
  Require(q4_first->byte_offset == 0 && q4_first->byte_size == 2 * 2 * 144,
          "Q4_K first range");
  Require(q4_second->byte_offset == 2 * 2 * 144 &&
              q4_second->byte_size == 2 * 2 * 144,
          "Q4_K second range");

  const TensorRef q8{data, GgmlType::kQ8_0, 32, 2, 4, 0, 0, "q8"};
  const auto q8_second = LocalExpertRange(q8, *small_second, &error);
  Require(q8_second.has_value(), error.c_str());
  Require(q8_second->byte_offset == 2 * 2 * 34 &&
              q8_second->byte_size == 2 * 2 * 34,
          "Q8_0 second range");

  const TensorRef mismatch{data, GgmlType::kQ8_0, 32, 2, 3, 0, 0, "bad"};
  Require(!LocalExpertRange(mismatch, *small_first, &error).has_value(),
          "reject mismatched expert count");
  auto invalid = *small_first;
  invalid.expert_count = 1;
  Require(!LocalExpertRange(q8, invalid, &error).has_value(),
          "reject malformed partition");
  RequireError(TpPartition::Create(3, 0, 2, &error),
               "reject uneven expert partition");
  RequireError(TpPartition::Create(0, 0, 1, &error),
               "reject empty expert partition");
  RequireError(TpPartition::Create(4, 2, 2, &error),
               "reject out-of-range rank");

  std::puts("PASS: TP=2 expert ownership and Q4_K/Q8_0 ranges");
  return 0;
}

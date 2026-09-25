// Dumps trunk prefill logits and optionally compares them with the F32 oracle.
// Use the production CLI for generation and the session test for replay.
//
// gpu_probe --model FIRST_SHARD.gguf --prompt TEXT [--reference]
//           [--batch T] [--context N] [--dump logits.bin]
// TP=2 probe: --tp-rank 0|1 --tp-world-size 2
//              --tp-bootstrap-host HOST --tp-bootstrap-port PORT
//              --tp-operation-id N (same on both ranks)
// Independent predictor oracle: --mtp-model MTP.gguf --mtp-audit
// MTP cost calibration: --mtp-model MTP.gguf --cost-audit C (0 = all)
//                      [--depth N] (default: 0, 4096, 32768)
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/distributed/tp_partition.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/communicator.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#ifdef GUFO_ENABLE_TP2_RDMA
#include "src/models/qwen38_flash_next/kernels/rocm/verbs.hpp"
#endif
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/reference.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace q = gufo::models::qwen38_flash_next;

void AuditMtp(q::rocm::Executor& executor, const q::rocm::DeviceModel& device,
              const q::ModelWeights& weights, const q::MtpWeights& mtp,
              const gufo::tokenization::QwenTokenizer& tokenizer,
              const std::filesystem::path& path);
void AuditMtpCosts(q::rocm::Executor& executor,
                   const gufo::tokenization::QwenTokenizer& tokenizer,
                   std::span<const std::uint32_t> depths,
                   std::uint32_t concurrency);

namespace {

#ifdef GUFO_ENABLE_TP2_RDMA
class OperationGuard {
public:
  OperationGuard(std::shared_ptr<q::rocm::Communicator> communicator,
                 std::uint64_t scope_id)
      : communicator_(std::move(communicator)), scope_id_(scope_id) {}

  ~OperationGuard() {
    if (!active_) {
      return;
    }
    try {
      std::string error;
      (void)communicator_->EndOperation(scope_id_, &error);
    } catch (...) {
    }
  }

  [[nodiscard]] bool Begin(std::string* error) {
    if (!communicator_->BeginOperation(scope_id_, error)) {
      return false;
    }
    active_ = true;
    return true;
  }

private:
  std::shared_ptr<q::rocm::Communicator> communicator_;
  const std::uint64_t scope_id_;
  bool active_{false};
};
#endif

struct Stats {
  std::uint32_t argmax_a;
  std::uint32_t argmax_b;
  double kl;
  double max_abs;
};

Stats Compare(const float* a, const float* b, std::size_t n) {
  std::vector<double> pa(n);
  std::vector<double> pb(n);
  const float ma = *std::max_element(a, a + n);
  const float mb = *std::max_element(b, b + n);
  double sa = 0.0;
  double sb = 0.0;
  double max_abs = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    pa[i] = std::exp(static_cast<double>(a[i] - ma));
    pb[i] = std::exp(static_cast<double>(b[i] - mb));
    sa += pa[i];
    sb += pb[i];
    max_abs = std::max(max_abs, std::fabs(static_cast<double>(a[i] - b[i])));
  }
  double kl = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    pa[i] /= sa;
    pb[i] /= sb;
    kl += pa[i] * (std::log(pa[i] + 1e-30) - std::log(pb[i] + 1e-30));
  }
  return {static_cast<std::uint32_t>(std::max_element(a, a + n) - a),
          static_cast<std::uint32_t>(std::max_element(b, b + n) - b), kl,
          max_abs};
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string prompt = "The capital of France is";
  std::string dump_path;
  std::string mtp_path;
  bool mtp_audit = false;
  bool reference = false;
  bool cost_audit = false;
  std::uint32_t cost_concurrency = 0;
  std::optional<std::uint32_t> cost_depth;
  std::uint32_t batch = 512;
  std::uint32_t context = 4096;
  std::uint32_t tp_rank = 0;
  std::uint32_t tp_world_size = 1;
  std::string tp_bootstrap_host;
  std::uint16_t tp_bootstrap_port = 18515;
  std::uint32_t tp_device = 0;
  std::uint32_t tp_gid = 0;
  std::uint64_t tp_operation_id = 1;
#ifndef GUFO_ENABLE_TP2_RDMA
  (void)tp_operation_id;
#endif
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : std::string();
    };
    if (arg == "--model") {
      model_path = next();
    } else if (arg == "--mtp-model") {
      mtp_path = next();
    } else if (arg == "--mtp-audit") {
      mtp_audit = true;
    } else if (arg == "--cost-audit") {
      cost_audit = true;
      const auto value = next();
      const auto [end, ec] = std::from_chars(
          value.data(), value.data() + value.size(), cost_concurrency);
      if (ec != std::errc{} || end != value.data() + value.size() ||
          (cost_concurrency != 0 && cost_concurrency != 1 &&
           cost_concurrency != 2 && cost_concurrency != 4 &&
           cost_concurrency != 6 && cost_concurrency != 8)) {
        std::fprintf(stderr, "--cost-audit requires C=0/1/2/4/6/8 (0 = all)\n");
        return 2;
      }
    } else if (arg == "--depth") {
      const auto value = next();
      std::uint32_t depth = 0;
      const auto [end, ec] =
          std::from_chars(value.data(), value.data() + value.size(), depth);
      if (ec != std::errc{} || end != value.data() + value.size()) {
        std::fprintf(stderr, "--depth requires a nonnegative integer\n");
        return 2;
      }
      cost_depth = depth;
    } else if (arg == "--tp-rank" || arg == "--tp-world-size" ||
               arg == "--tp-bootstrap-port" || arg == "--tp-device" ||
               arg == "--tp-gid-index") {
      const auto value = next();
      std::uint32_t parsed = 0;
      const auto [end, ec] =
          std::from_chars(value.data(), value.data() + value.size(), parsed);
      if (ec != std::errc{} || end != value.data() + value.size()) {
        std::fprintf(stderr, "%s requires an unsigned integer\n", arg.c_str());
        return 2;
      }
      if (arg == "--tp-rank") {
        tp_rank = parsed;
      } else if (arg == "--tp-world-size") {
        tp_world_size = parsed;
      } else if (arg == "--tp-bootstrap-port") {
        if (parsed == 0 || parsed > std::numeric_limits<std::uint16_t>::max()) {
          std::fprintf(stderr, "--tp-bootstrap-port is out of range\n");
          return 2;
        }
        tp_bootstrap_port = static_cast<std::uint16_t>(parsed);
      } else if (arg == "--tp-device") {
        if (parsed >
            static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
          std::fprintf(stderr, "--tp-device is out of range\n");
          return 2;
        }
        tp_device = parsed;
      } else {
        tp_gid = parsed;
      }
    } else if (arg == "--tp-operation-id") {
      const auto value = next();
      const auto [end, ec] = std::from_chars(
          value.data(), value.data() + value.size(), tp_operation_id);
      if (ec != std::errc{} || end != value.data() + value.size()) {
        std::fprintf(stderr,
                     "--tp-operation-id requires an unsigned integer\n");
        return 2;
      }
    } else if (arg == "--tp-bootstrap-host") {
      tp_bootstrap_host = next();
    } else if (arg == "--prompt") {
      prompt = next();
    } else if (arg == "--reference") {
      reference = true;
    } else if (arg == "--batch" || arg == "--context") {
      const auto value = next();
      auto& count = arg == "--batch" ? batch : context;
      const auto [end, ec] =
          std::from_chars(value.data(), value.data() + value.size(), count);
      if (ec != std::errc{} || end != value.data() + value.size() ||
          count == 0) {
        std::fprintf(stderr, "%s requires a positive integer\n", arg.c_str());
        return 2;
      }
    } else if (arg == "--dump") {
      dump_path = next();
    } else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  if (model_path.empty()) {
    std::fprintf(stderr, "--model is required\n");
    return 2;
  }
  if ((mtp_audit || cost_audit) && mtp_path.empty()) {
    std::fprintf(stderr, "MTP audits require --mtp-model\n");
    return 2;
  }
  if (cost_audit && (mtp_audit || reference || !dump_path.empty())) {
    std::fprintf(stderr, "--cost-audit cannot be combined with other probes\n");
    return 2;
  }
  if (cost_depth && !cost_audit) {
    std::fprintf(stderr, "--depth requires --cost-audit\n");
    return 2;
  }
  if (tp_world_size != 1 && tp_world_size != 2) {
    std::fprintf(stderr, "--tp-world-size must be 1 or 2\n");
    return 2;
  }
  if (tp_world_size == 1 && tp_rank != 0) {
    std::fprintf(stderr, "single-rank probe requires --tp-rank 0\n");
    return 2;
  }
  if (tp_rank >= tp_world_size) {
    std::fprintf(stderr, "--tp-rank must be less than --tp-world-size\n");
    return 2;
  }
  if (tp_world_size == 2 && tp_rank == 1 && tp_bootstrap_host.empty()) {
    std::fprintf(stderr, "rank one requires --tp-bootstrap-host\n");
    return 2;
  }
  if (tp_device > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      hipSetDevice(static_cast<int>(tp_device)) != hipSuccess) {
    std::fprintf(stderr, "HIP device selection failed\n");
    return 1;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return 1;
  }
  auto weights = q::ModelWeights::Bind(*reader, &error);
  if (!weights) {
    std::fprintf(stderr, "bind failed: %s\n", error.c_str());
    return 1;
  }
  auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (!tokenizer) {
    std::fprintf(stderr, "tokenizer failed: %s\n", error.c_str());
    return 1;
  }
  const auto& c = weights->config;
  if (cost_depth &&
      (c.context_length < 96 || *cost_depth > c.context_length - 96)) {
    std::fprintf(stderr, "--depth leaves no room for the cost-audit suffix\n");
    return 2;
  }
  std::optional<q::distributed::TpPartition> partition;
  std::shared_ptr<q::rocm::Communicator> communicator;
#ifdef GUFO_ENABLE_TP2_RDMA
  std::unique_ptr<OperationGuard> operation;
#endif
  if (tp_world_size == 2) {
    partition = q::distributed::TpPartition::Create(c.num_experts, tp_rank,
                                                    tp_world_size, &error);
    if (!partition) {
      std::fprintf(stderr, "partition failed: %s\n", error.c_str());
      return 1;
    }
#ifdef GUFO_ENABLE_TP2_RDMA
    const q::rocm::IbrverbsConfig config{
        .rank = tp_rank,
        .world_size = tp_world_size,
        .bootstrap_host = tp_bootstrap_host,
        .bootstrap_port = tp_bootstrap_port,
        .device_index = tp_device,
        .gid_index = tp_gid,
    };
    communicator = q::rocm::CreateIbrverbsCommunicator(config, &error);
    if (!communicator) {
      std::fprintf(stderr, "RDMA communicator failed: %s\n", error.c_str());
      return 1;
    }
    operation = std::make_unique<OperationGuard>(communicator, tp_operation_id);
    if (!operation->Begin(&error)) {
      std::fprintf(stderr, "TP operation scope bind failed: %s\n",
                   error.c_str());
      return 1;
    }
#else
    std::fprintf(stderr, "this probe was built without TP2 RDMA support\n");
    return 2;
#endif
  }
  std::unique_ptr<q::NgramTable> ngram;
  if (c.ple_layer >= 0) {
    const auto& t = weights->ple_table;
    ngram = q::NgramTable::Open(
        reader->GetMappedRegions()[t.shard].file_descriptor, t.file_offset,
        t.rows, c.ple_head_dim, t.type, &error);
    if (!ngram) {
      std::fprintf(stderr, "n-gram table failed: %s\n", error.c_str());
      return 1;
    }
  }

  std::shared_ptr<gufo::core::GgufReader> mtp_reader;
  std::optional<q::MtpWeights> mtp_weights;
  if (!mtp_path.empty()) {
    mtp_reader = gufo::core::GgufReader::OpenFile(mtp_path, &error);
    if (mtp_reader)
      mtp_weights = q::MtpWeights::Bind(*mtp_reader, c, &error);
    if (!mtp_weights) {
      std::fprintf(stderr, "MTP bind failed: %s\n", error.c_str());
      return 1;
    }
  }
  if (partition) {
    const auto routed_plan = q::distributed::PlanRoutedBytes(
        *weights, *partition, mtp_weights ? &*mtp_weights : nullptr, &error);
    if (!routed_plan) {
      std::fprintf(stderr, "routed weight plan failed: %s\n", error.c_str());
      return 1;
    }
    std::printf("TP2 routed_bytes full=%zu local=%zu\\n",
                routed_plan->full_encoded_bytes,
                routed_plan->local_encoded_bytes);
  }
  const auto* partition_ptr = partition ? &*partition : nullptr;
  auto device = q::rocm::DeviceModel::Upload(
      *weights, *reader, mtp_weights ? &*mtp_weights : nullptr,
      mtp_reader.get(), &error, partition_ptr);
  if (!device) {
    std::fprintf(stderr, "upload failed: %s\n", error.c_str());
    return 1;
  }
  if (tp_world_size == 2) {
    std::printf("TP2 rank=%u world=%u local_experts=%u resident_bytes=%zu\n",
                tp_rank, tp_world_size, device->local_experts(),
                device->resident_bytes());
  }
  q::rocm::Executor::Options options;
  options.device_index = static_cast<int>(tp_device);
  options.max_batch = batch;
  options.max_logit_rows = std::min<std::uint32_t>(batch, 64);
  if (mtp_audit || cost_audit)
    options.max_speculative = 8;
  if (communicator) {
    options.all_reduce = [communicator](float* data, std::size_t bytes,
                                        hipStream_t stream,
                                        std::string* error) {
      return communicator->AllReduceSum(data, bytes, stream, error);
    };
  }

  if (cost_audit) {
    options.max_batch = 2048;
    options.max_logit_rows = 8;
  }
  auto executor =
      q::rocm::Executor::Create(*device, ngram.get(), options, &error);
  if (!executor) {
    std::fprintf(stderr, "executor failed: %s\n", error.c_str());
    return 1;
  }
  if (cost_audit) {
    try {
      const std::array<std::uint32_t, 3> depths{0, 4096, 32768};
      std::span<const std::uint32_t> selected_depths = depths;
      if (cost_depth)
        selected_depths = std::span(&*cost_depth, 1);
      AuditMtpCosts(*executor, *tokenizer, selected_depths, cost_concurrency);
      return 0;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "cost audit failed: %s\n", e.what());
      return 1;
    }
  }
  if (mtp_audit) {
    try {
      AuditMtp(*executor, *device, *weights, *mtp_weights, *tokenizer,
               model_path);
      return 0;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "MTP oracle failed: %s\n", e.what());
      return 1;
    }
  }
  auto session = executor->CreateSession(
      gufo::core::SessionMode::kAutoregressive, context, &error);
  if (!session) {
    std::fprintf(stderr, "session failed: %s\n", error.c_str());
    return 1;
  }

  std::vector<std::int32_t> tokens;
  for (auto id : tokenizer->Encode(prompt)) {
    tokens.push_back(static_cast<std::int32_t>(id));
  }
  if (tokens.empty() || tokens.size() > context) {
    std::fprintf(stderr, "prompt must contain 1..%u tokens\n", context);
    return 2;
  }
  std::printf("prompt tokens (%zu)\n", tokens.size());
  std::ofstream dump;
  if (!dump_path.empty()) {
    dump.open(dump_path, std::ios::binary);
    if (!dump) {
      std::fprintf(stderr, "cannot open logit dump %s\n", dump_path.c_str());
      return 1;
    }
  }

  std::vector<float> gpu_logits;
  std::size_t logit_rows = 0;
  for (std::size_t off = 0; off < tokens.size(); off += batch) {
    const std::size_t n = std::min<std::size_t>(batch, tokens.size() - off);
    const auto rows = static_cast<std::uint32_t>(
        std::min<std::size_t>(n, options.max_logit_rows));
    std::vector<float> out(static_cast<std::size_t>(rows) * c.vocab_size);
    if (!executor->Forward(
            *session, std::span<const std::int32_t>(tokens.data() + off, n),
            rows, out.data(), q::rocm::Executor::ForwardMode::kPrefill,
            &error)) {
      std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
      return 1;
    }
    const bool last = off + n == tokens.size();
    if (last) {
      gpu_logits = std::move(out);
      logit_rows = rows;
    }
  }
  if (dump.is_open()) {
    dump.write(reinterpret_cast<const char*>(gpu_logits.data()),
               static_cast<std::streamsize>(gpu_logits.size() * sizeof(float)));
    dump.flush();
    if (!dump) {
      std::fprintf(stderr, "cannot write logit dump %s\n", dump_path.c_str());
      return 1;
    }
  }

  if (reference) {
    q::ReferenceModel ref(*weights, ngram.get(),
                          static_cast<std::uint32_t>(tokens.size()));
    std::vector<float> ref_logits(c.vocab_size);
    const std::size_t first = tokens.size() - logit_rows;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      if (!ref.Step(tokens[i], ref_logits, {}, &error)) {
        std::fprintf(stderr, "reference failed: %s\n", error.c_str());
        return 1;
      }
      if (i < first) {
        continue;
      }
      const Stats st =
          Compare(ref_logits.data(),
                  gpu_logits.data() + (i - first) * c.vocab_size, c.vocab_size);
      std::printf("pos %3zu ref %6u gpu %6u %s  KL %.5f  max|d| %.3f\n", i,
                  st.argmax_a, st.argmax_b,
                  st.argmax_a == st.argmax_b ? "OK  " : "DIFF", st.kl,
                  st.max_abs);
    }
  }

  return 0;
}

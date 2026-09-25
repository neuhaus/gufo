// PLE n-gram gather fingerprint for the full-Q8 long-context divergence
// investigation. Host-only: no ROCm device is created, and nothing here runs
// the model, so a run needs only the GGUF files and a filesystem.
//
// The tool deliberately opens its own `NgramTable` instead of reaching into
// the engine (`Model::ngram_` is private): the point is to exercise
// `NgramTable::Open` and the disk read path directly, because that is what
// could differ between two hosts whose shards already hash identically.
//
// Two digests are printed separately and never combined:
//
//   * `rows.sha256`    - the gathered row-id list. A mismatch here is a
//                        tokenization or history-advance difference, and it
//                        exonerates the table contents.
//   * `content_sha256` - the bit pattern of the dequantized F32 row bytes. A
//                        mismatch here with identical row ids is a table
//                        read/dequantize difference, so the PLE hypothesis is
//                        still live.
//
//   ple_gather_probe [--model SHARD.gguf]
//                    [--prompt TEXT | --prompt-file PATH]
//                    [--repeat N] [--async] [--help]
//
// stdout is the compared surface: `key value` lines with no timestamps, paths,
// pids, timings or hostnames. Phase notes and the resolved shard file name go
// to stderr. stdout is flushed at exit, so capture it on its own
// (`2>/dev/null`) rather than merging the two streams.
//
// Exit codes: 0 success, 1 runtime failure, 2 usage error.

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace q = gufo::models::qwen38_flash_next;

namespace {

// Production path on both hosts (docs/models/qwen3.8-flash-next/Q8.md).
constexpr std::string_view kDefaultModel =
    "/opt/models/qwen3.8-flash-next/Q8_0/"
    "Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf";

constexpr int kExitOk = 0;
constexpr int kExitRuntime = 1;
constexpr int kExitUsage = 2;

void Note(const std::string& message) {
  std::fputs(("ple_gather_probe: " + message + "\n").c_str(), stderr);
}

std::string Format(const char* fmt, ...) {
  char buffer[256];
  va_list args;
  va_start(args, fmt);
  const int written = std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  if (written <= 0) {
    return std::string();
  }
  const std::size_t size = static_cast<std::size_t>(written);
  return std::string(buffer,
                     size < sizeof(buffer) - 1 ? size : sizeof(buffer) - 1);
}

double ElapsedMs(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

void Usage(std::FILE* out) {
  std::fprintf(
      out,
      "usage: ple_gather_probe [--model SHARD.gguf]\n"
      "                       [--prompt TEXT | --prompt-file PATH]\n"
      "                       [--repeat N] [--async] [--help]\n"
      "\n"
      "  --model PATH       GGUF shard of the split set that carries\n"
      "                     per_layer_token_embd.weight. Any shard of the\n"
      "                     set works: GgufReader::OpenFile assembles the\n"
      "                     whole set and the owning region is taken from\n"
      "                     ple_table.shard. Default, the production Q8 path:\n"
      "                     %s\n"
      "  --prompt TEXT      Prompt text. Mutually exclusive with\n"
      "                     --prompt-file.\n"
      "  --prompt-file PATH Read the prompt from a file, byte for byte.\n"
      "                     Mutually exclusive with --prompt.\n"
      "  --repeat N         Gather passes in one process, default 3. The\n"
      "                     first pass is cold, later ones may hit the\n"
      "                     resident row cache. One extra cross-check pass\n"
      "                     always runs on the other read path.\n"
      "  --async            Gather with StartRead/WaitRead instead of Read.\n"
      "  --help             This text.\n"
      "\n"
      "exit codes: 0 success, 1 runtime failure, 2 usage error\n",
      std::string(kDefaultModel).c_str());
}

// std::as_bytes yields a byte span; the hasher takes uint8_t, so every digest
// below goes through this one reinterpretation and nowhere else.
template<typename T>
std::span<const std::uint8_t> BytesOf(std::span<const T> values) {
  return {reinterpret_cast<const std::uint8_t*>(values.data()),
          values.size_bytes()};
}

// Why the digests are over bit patterns rather than decimal renderings: both
// hosts run the same deterministic decoder over the same file bytes, so the
// only question is whether the produced bits are identical. A decimal
// rendering would round, hiding bit differences that still change downstream
// arithmetic; it would collapse -0.0 and 0.0 and every NaN payload onto the
// same text; and it would depend on locale and printf formatting. Hashing raw
// bytes therefore never needs NaN to be absent, and never needs a canonical
// float comparison.
//
// What bitwise hashing catches: any difference in the gathered row bytes -
// a wrong row, a misaligned or short read, a stale cache entry, a decoder
// that differs by one ULP.
//
// What it does not catch: an error both hosts make identically. This is a
// cross-host differential, not a correctness oracle. Matching digests show
// the PLE read path is not what differs between the two hosts; they do not
// show the values are right, and they say nothing about the router, which
// this tool never runs.
std::string HashFloats(std::span<const float> values) {
  return gufo::crypto::Sha256Hex(BytesOf(values));
}

// Row ids are hashed as raw uint32 bytes. Both hosts are x86-64, so the byte
// order is fixed and the id list is compared exactly, not through a rendering
// that could hide a single differing id.
std::string HashRows(std::span<const std::uint32_t> rows) {
  gufo::crypto::Sha256Hasher hasher;
  hasher.Update(BytesOf(rows));
  return hasher.FinishHex();
}

std::string HashTokens(std::span<const std::int32_t> tokens) {
  gufo::crypto::Sha256Hasher hasher;
  hasher.Update(BytesOf(tokens));
  return hasher.FinishHex();
}

bool ReadFileBytes(const std::string& path, std::string* out,
                   std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open prompt file: " + path;
    return false;
  }
  out->assign(std::istreambuf_iterator<char>(in),
              std::istreambuf_iterator<char>());
  if (in.bad()) {
    *error = "read error on prompt file: " + path;
    return false;
  }
  return true;
}

bool ParseCount(const std::string& text, long long* out) {
  if (text.empty()) {
    return false;
  }
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), *out);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string prompt;
  std::string prompt_file;
  long long repeat = 3;
  bool model_explicit = false;
  bool prompt_set = false;
  bool prompt_file_set = false;
  bool async = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      Usage(stdout);
      return kExitOk;
    }
    if (arg == "--async") {
      async = true;
      continue;
    }
    if (arg == "--model" || arg == "--prompt" || arg == "--prompt-file" ||
        arg == "--repeat") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", arg.c_str());
        return kExitUsage;
      }
      const std::string value = argv[++i];
      if (arg == "--model") {
        model_path = value;
        model_explicit = true;
      } else if (arg == "--prompt") {
        prompt = value;
        prompt_set = true;
      } else if (arg == "--prompt-file") {
        prompt_file = value;
        prompt_file_set = true;
      } else if (!ParseCount(value, &repeat)) {
        std::fprintf(stderr, "--repeat needs a non-negative integer\n");
        return kExitUsage;
      }
      continue;
    }
    std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
    Usage(stderr);
    return kExitUsage;
  }

  // Every usage check runs before the model or the table is touched, so a bad
  // invocation can never open the ~54 GB PLE payload.
  if (model_explicit && model_path.empty()) {
    std::fprintf(stderr, "--model path is empty\n");
    return kExitUsage;
  }
  if (prompt_set && prompt_file_set) {
    std::fprintf(stderr, "--prompt and --prompt-file are mutually exclusive\n");
    return kExitUsage;
  }
  if (!prompt_set && !prompt_file_set) {
    std::fprintf(stderr, "one of --prompt or --prompt-file is required\n");
    return kExitUsage;
  }
  if (repeat <= 0) {
    std::fprintf(stderr, "--repeat must be at least 1\n");
    return kExitUsage;
  }
  if (model_path.empty()) {
    model_path = std::string(kDefaultModel);
  }
  if (prompt_file_set) {
    std::string error;
    if (!ReadFileBytes(prompt_file, &prompt, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return kExitUsage;
    }
    if (prompt.empty()) {
      std::fprintf(stderr, "prompt file is empty: %s\n", prompt_file.c_str());
      return kExitUsage;
    }
  } else if (prompt.empty()) {
    std::fprintf(stderr, "--prompt text is empty\n");
    return kExitUsage;
  }

  std::string error;
  const auto open_start = std::chrono::steady_clock::now();
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return kExitRuntime;
  }
  Note(Format("opened %zu mapped region(s) in %.0f ms",
              reader->GetMappedRegions().size(), ElapsedMs(open_start)));

  // ple_table.shard is the index of the mapped region that owns the tensor, and
  // the split set numbers regions by their own `split.no`. The full-Q8 set
  // declares per_layer_token_embd.weight in split.no 2, the third file.
  auto weights = q::ModelWeights::Bind(*reader, &error);
  if (!weights) {
    std::fprintf(stderr, "bind failed: %s\n", error.c_str());
    return kExitRuntime;
  }
  const q::Config& c = weights->config;
  if (c.ple_layer < 0) {
    std::fprintf(stderr, "model carries no PLE table\n");
    return kExitRuntime;
  }
  const q::TensorRef& table = weights->ple_table;
  if (table.empty()) {
    std::fprintf(stderr, "per_layer_token_embd.weight is missing or unbound\n");
    return kExitRuntime;
  }
  const auto regions = reader->GetMappedRegions();
  if (table.shard >= regions.size()) {
    std::fprintf(stderr, "PLE shard %u is outside the %zu mapped region(s)\n",
                 table.shard, regions.size());
    return kExitRuntime;
  }
  Note(Format(
      "PLE tensor is in mapped region %u of %zu, split file %05u-of-%05u",
      table.shard, regions.size(), table.shard + 1,
      static_cast<unsigned>(regions.size())));
  if (c.ple_heads == 0 || c.ple_head_dim == 0) {
    std::fprintf(stderr, "PLE geometry is empty (heads %u, dim %u)\n",
                 c.ple_heads, c.ple_head_dim);
    return kExitRuntime;
  }

  // Mirrors src/models/qwen38_flash_next/engine.cpp:113-121: the bound
  // descriptor of the owning region, the tensor's offset inside it, the row
  // count taken from the file, ple_head_dim and the tensor's own type.
  const auto table_start = std::chrono::steady_clock::now();
  auto ngram = q::NgramTable::Open(regions[table.shard].file_descriptor,
                                   table.file_offset, table.rows,
                                   c.ple_head_dim, table.type, &error);
  if (!ngram) {
    std::fprintf(stderr, "n-gram table failed: %s\n", error.c_str());
    return kExitRuntime;
  }
  Note(Format("opened the PLE table in %.0f ms", ElapsedMs(table_start)));

  auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (!tokenizer) {
    std::fprintf(stderr, "tokenizer failed: %s\n", error.c_str());
    return kExitRuntime;
  }
  const auto tokenize_start = std::chrono::steady_clock::now();
  const std::vector<gufo::tokenization::TokenId> encoded =
      tokenizer->Encode(prompt);
  std::vector<std::int32_t> tokens;
  tokens.reserve(encoded.size());
  for (const auto id : encoded) {
    tokens.push_back(static_cast<std::int32_t>(id));
  }
  Note(Format("tokenized %zu byte(s) into %zu token(s) in %.0f ms",
              prompt.size(), tokens.size(), ElapsedMs(tokenize_start)));
  if (tokens.empty()) {
    std::fprintf(stderr, "prompt tokenized to zero tokens\n");
    return kExitRuntime;
  }

  // One call over the whole prompt: `history` advances past every token, so an
  // EOS anywhere in the window cuts older context and a missing predecessor
  // reads as EOS exactly as the per-token decode path sees it.
  std::vector<std::uint32_t> rows(tokens.size() * c.ple_heads);
  const auto hash_start = std::chrono::steady_clock::now();
  q::NgramHistory history;
  q::HashNgramRows(c, history, tokens, rows);
  Note(Format("hashed %zu row id(s) in %.0f ms", rows.size(),
              ElapsedMs(hash_start)));

  for (const std::uint32_t row : rows) {
    if (row >= ngram->Rows()) {
      std::fprintf(stderr,
                   "row id %u is outside the table's %llu rows; PLE geometry "
                   "and table disagree\n",
                   row, static_cast<unsigned long long>(ngram->Rows()));
      return kExitRuntime;
    }
  }
  const std::string rows_hash = HashRows(rows);
  std::vector<std::uint32_t> sorted(rows);
  std::sort(sorted.begin(), sorted.end());
  const auto distinct = static_cast<std::size_t>(
      std::unique(sorted.begin(), sorted.end()) - sorted.begin());
  const std::string unique_hash =
      HashRows(std::span<const std::uint32_t>(sorted.data(), distinct));
  const std::string tokens_hash = HashTokens(tokens);

  std::vector<float> gathered(rows.size() * ngram->RowDim());
  std::printf("ple-gather-probe 1\n");
  std::printf("prompt.tokens %zu\n", tokens.size());
  std::printf("prompt.sha256 %s\n", tokens_hash.c_str());
  std::printf("table.tensor %s\n", std::string(table.name).c_str());
  std::printf("table.type %s\n",
              std::string(gufo::core::ToString(table.type)).c_str());
  std::printf("table.row_dim %u\n", ngram->RowDim());
  std::printf("table.rows %llu\n",
              static_cast<unsigned long long>(ngram->Rows()));
  std::printf("table.row_bytes %zu\n", ngram->RowBytes());
  std::printf("table.shard %u\n", table.shard);
  std::printf("table.shard_count %zu\n", regions.size());
  std::printf("rows.total %zu\n", rows.size());
  std::printf("rows.distinct %zu\n", distinct);
  std::printf("rows.in_range yes\n");
  std::printf("rows.sha256 %s\n", rows_hash.c_str());
  std::printf("rows.unique.sha256 %s\n", unique_hash.c_str());
  std::printf("gather.floats %zu\n", gathered.size());
  std::printf("gather.bytes %zu\n", gathered.size() * sizeof(float));

  // One gather per pass. The first is cold, later ones may hit the resident
  // row cache. Every pass gathers the whole row set, so a short or failed read
  // leaves part of the buffer unwritten and must never reach a digest.
  const auto gather = [&](const char* label, bool use_async) -> std::string {
    const char* path = use_async ? "async" : "read";
    const auto start = std::chrono::steady_clock::now();
    const bool ok =
        use_async ? (ngram->StartRead(rows, gathered) && ngram->WaitRead())
                  : ngram->Read(rows, gathered);
    Note(Format("pass %s via %s %s in %.0f ms", label, path,
                ok ? "ok" : "FAILED", ElapsedMs(start)));
    if (!ok) {
      std::fprintf(stderr, "gather failed on pass %s (%s path)\n", label, path);
      return std::string();
    }
    const std::string digest = HashFloats(gathered);
    std::printf("pass %s path %s content_sha256 %s\n", label, path,
                digest.c_str());
    return digest;
  };

  std::string first_hash;
  for (long long i = 0; i < repeat; ++i) {
    const std::string label = std::to_string(i);
    const std::string digest = gather(label.c_str(), async);
    if (digest.empty()) {
      return kExitRuntime;
    }
    if (i == 0) {
      first_hash = digest;
    } else if (digest != first_hash) {
      std::fprintf(stderr,
                   "intra-run mismatch: pass %s gathered different bytes than "
                   "pass 0\n",
                   label.c_str());
      std::printf("passes_identical no\n");
      std::printf("result mismatch\n");
      std::fflush(stdout);
      return kExitRuntime;
    }
  }
  // StartRead/WaitRead is a separate code path from Read, so run one pass on
  // the other one and compare: a single invocation covers both.
  const std::string cross = gather("cross", !async);
  if (cross.empty()) {
    return kExitRuntime;
  }
  if (cross != first_hash) {
    std::fprintf(
        stderr,
        "intra-run mismatch: the %s path gathered different bytes than "
        "the %s path\n",
        async ? "read" : "async", async ? "async" : "read");
    std::printf("passes_identical no\n");
    std::printf("result mismatch\n");
    std::fflush(stdout);
    return kExitRuntime;
  }
  std::printf("passes_identical yes\n");
  std::printf("result ok\n");
  std::fflush(stdout);
  return kExitOk;
}

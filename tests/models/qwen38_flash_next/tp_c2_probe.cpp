// Dormant C2 cohort probe: one binary, two roles, two hosts.
//
// qwen38_flash_next_tp_c2_probe --role rank0   (this file; RDMA + control peer,
//                                               model, lockstep driver)
// qwen38_flash_next_tp_c2_probe --role rank1   (also this file; the real
//                                               InferenceBackend C2 worker,
//                                               hosted in-process)
//
// WHAT THIS PROVES
//   * the RDMA peer, the TP control channel and the C2 handshake complete;
//   * `SubmitCohort` admits a two-member `kCohort2Ar` command;
//   * the two members execute in command order inside ONE collective scope;
//   * each member's response tokens agree with rank 0's own greedy output.
//
// WHAT THIS PROVES NOTHING ABOUT
//   The C2 slice is dormant and serial by design (the rank-1 runner pool
//   capacity is 1, so member 0 finishes before member 1 is admitted). This
//   probe does not measure or validate physical/batched C2 execution. Nothing
//   here relaxes a production guard, enables `--sessions 2`, or adds a CLI
//   alias.
//
// COLLECTIVE TRACE ARITHMETIC (load-bearing)
//   One `Executor::Forward(n)` runs the trunk layer loop once and issues
//   exactly `num_layers` `AllReduceSum` collectives of `n * hidden_size * 4`
//   bytes (executor.cpp: one `Moe` -> one `AllReduce` per layer).
//
//   The worker runs each cohort member as independent serial C1 work, so for
//   member m with prompt P and budget K, without MTP:
//     prefill: one `Sync(P)`  -> 1 Forward(n=|P|)   (a single chunk requires
//              |P| <= PrefillCapacity(); this probe refuses longer prompts)
//     decode:  the scheduler alternates `SelectNext()` (samples the current
//              logits, NO forward) with `Advance()` (= `Evaluate` = 1 Forward
//              of one row). `TextRunnerCapabilities::final_token_advance_-
//              required` is false for the distributed Flash-Next runner, so
//              `PrepareDecode` completes the request at the token limit
//              WITHOUT advancing the final token.
//     => tokens published T, forwards A = T - 1 when the run ends on the
//        length limit, and A = T when `SelectNext` returns a stop token first.
//        Per member: 1 + A forwards, i.e. (1 + A) * num_layers collectives.
//
//   `RunMember` below reproduces exactly that: sample from `session->Logits()`
//   (no collective), stop on `model->IsStopToken`, stop at the budget, and only
//   then `session->Evaluate(token)` (one collective group). It deliberately
//   does NOT use `Session::DecodeStep`, which always advances the token it
//   samples and would therefore issue one extra forward per member and
//   desynchronise the trace.
//
// OPERATING RULES
//   * Start BOTH roles under an external timeout (`timeout 900 ...`). The RDMA
//     per-collective timeout is 30 s, but the TP control response wait uses a
//     24 h socket timeout, so a dead rank-1 peer can hang rank 0 here.
//   * The RDMA path maps FIXED IOVA windows (verbs.cpp: 64 MiB each at
//     0x700000000000, +0x4000000, +0x8000000). The two roles CANNOT share a
//     host with any other RDMA process.
//   * Rank 0 opens the RDMA bootstrap FIRST and the control listener SECOND.
//     Both bootstraps block for 30 s and would deadlock in the other order.
//     `--role rank1` mirrors the production order in serve.cpp: RDMA connect
//     first, control connect second, handshake third
//     (`InferenceBackend::load`). Both sides must be started within 30 s.
//   * Rank 0 sends the cohort the moment its own model is loaded, so rank 1
//     must finish loading before rank 0's first prefill collective, which also
//     times out after 30 s. `--role rank1` logs its ready line at that point.
//   * `gufo serve` cannot host the rank-1 worker: it forces
//     `--max-pending 1` while `SubmitCohort` needs `queued + 2 <= max_pending`,
//     so a cohort is always refused at admission. `--role rank1` hosts
//     `InferenceBackend` in-process with `max_pending_requests = 2`, runner
//     pool capacity 1, no MTP, and `--context` equal to rank 0's.
//   * Rank 1 never validates the C2 contract; it reports what its worker loop
//     returns. The C2 verdict is always rank 0's exit code.
//
// EXIT REASONS
//   0  every C2 contract and token-agreement check passed (rank 0), or the
//      control channel closed after the cohort (rank 1)
//   1  transport/model/execution failure (RDMA, control, decode)
//   2  argument or pre-flight validation failure
//   3  reserved; retired with the unimplemented rank-1 stub
//   4  response envelope mismatch (sequence, kind or cohort id)
//   5  response plan-digest mismatch
//   6  response member count or member order mismatch
//   7  the worker answered with a C2 error response
//   8  member 0 returned more tokens than its budget
//   9  member 1 returned more tokens than its budget
//  10  member 0 tokens disagree with rank 0's local greedy output
//  11  member 1 tokens disagree with rank 0's local greedy output
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/inference_backend.hpp"
#include "src/cli/serve/tp_control.hpp"
#include "src/core/sampling.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/verbs.hpp"

namespace q = gufo::models::qwen38_flash_next;
namespace server = gufo::server;

namespace {

constexpr const char* kDefaultModel =
    "/opt/models/qwen3.8-flash-next/UD-Q4_K_XL/"
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";
/// C2 is fixed at two members (text_generation_scheduler.hpp).
constexpr std::size_t kMemberCount = 2;
/// TP=2 is fixed at two ranks (tp_control.cpp, verbs.cpp).
constexpr std::uint32_t kWorldSize = 2;
constexpr std::uint32_t kRank1 = 1;
/// The worker sends `TextPrefillPolicy::decode_active_tokens`; the handshake
/// compares it verbatim. The value does not chunk the collective trace at
/// runner capacity 1, where the only resident member prefills unbounded.
constexpr std::uint32_t kWorkerPrefillChunkTokens = 512;
/// `InferenceBackend::load` sizes the runner pool from the session count, and
/// the C2 response contract requires every member to report
/// `execution_plan == "serial-c1"` with `physical_execution_width == 1`
/// (tp_cohort_plan.cpp `IsSerialPlanMember`). Two runners could interleave the
/// members and break rank 0's lockstep collective arithmetic, so the worker
/// keeps the production capacity of exactly one.
constexpr std::size_t kWorkerSessionCount = 1;
/// `SubmitCohort` refuses admission unless `queued_count + 2 <= max_pending`
/// (text_generation_scheduler.cpp:1515-1519), and `gufo serve` hard-codes
/// `--max-pending 1` for TP=2. Two is the smallest value that admits a cohort;
/// it is the policy this probe supplies, not a production change.
constexpr std::size_t kWorkerMaxPendingRequests = 2;
/// Rank 0 sends one member per distinct client id, so the per-client ceiling
/// stays at 1 and each member is admitted on its own.
constexpr std::size_t kWorkerMaxPendingRequestsPerClient = 1;
/// `gufo_tp_control` bound (tp_control.cpp kMaxAuthTokenBytes).
constexpr std::size_t kMaxAuthTokenBytes = 4096;
/// `ValidateTpControlCommand` rejects a C2 member whose id is zero.
constexpr std::uint64_t kMinMemberId = 1;
/// The worker's C2 admission builds a default `SamplingConfig`
/// (tp_cohort_plan.cpp), and the C2 envelope carries no sampling fields, so
/// both roles decode greedily.
constexpr float kGreedyTemperature = 0.0F;
/// `run_worker` reports rank 0's exit as a closed channel (tp_control.cpp
/// `RecvAll`). That is the one worker stop that is not a rank-1 failure.
constexpr std::string_view kPeerClosedChannel =
    "TP control peer closed the channel";

enum ExitReason {
  kOk = 0,
  kTransportFailure = 1,
  kInvalidArguments = 2,
  /// Retired with the unimplemented rank-1 stub. The value stays reserved so
  /// the rank-zero reasons below keep their documented numbers.
  kReservedRetiredRank1 = 3,
  kEnvelopeMismatch = 4,
  kDigestMismatch = 5,
  kMemberOrderMismatch = 6,
  kCohortErrorResponse = 7,
  kMember0BudgetExceeded = 8,
  kMember1BudgetExceeded = 9,
  kMember0TokenMismatch = 10,
  kMember1TokenMismatch = 11,
};

void Say(const char* role, const std::string& line) {
  std::printf("[c2-probe %s] %s\n", role, line.c_str());
  std::fflush(stdout);
}

void Warn(const std::string& line) {
  std::fprintf(stderr, "[c2-probe] %s\n", line.c_str());
  std::fflush(stderr);
}

void Usage() {
  std::puts(R"(Dormant C2 (kCohort2Ar) two-host probe for TP=2 Flash-Next.

Usage:
  qwen38_flash_next_tp_c2_probe --role rank0 [options]
  qwen38_flash_next_tp_c2_probe --role rank1 --tp-bootstrap-host HOST [options]

Role:
  --role rank0|rank1      rank0 = RDMA peer, control peer, model and the
                          lockstep cohort driver. rank1 = the real
                          InferenceBackend C2 worker, hosted in-process: it has
                          no cohort options and only serves what rank 0 sends.

Rank-one options:
  --tp-bootstrap-host HOST  REQUIRED for rank1. Rank 0's address, used for the
                          RDMA bootstrap and for the TP control connect. rank0
                          ignores it and binds the bootstrap to all interfaces.

Peer options (both roles; the values must match or the handshake rejects):
  --model PATH            First GGUF shard (default: production Q4 path)
  --tp-bootstrap-port N   RDMA bootstrap port (default 18515)
  --tp-control-port N     TP control port (default 18516)
  --tp-control-token TOK  Nonempty token; must equal the peer's, <= 4096 B
  --tp-device N           HIP device index (default 0)
  --tp-gid-index N        RDMA GID index (default 0)
  --context N             Handshake context; must equal the peer's
                          (default 4096)
  --help                  This text

Rank-zero options (ignored by rank1):
  --cohort-id N           C2 cohort identity, nonzero (default 1)
  --sequence N            C2 sequence and the collective scope, nonzero
                          (default 1)
  --member-id-0 N         Member 0 id, nonzero (default 1)
  --member-id-1 N         Member 1 id, nonzero and distinct (default 2)
  --max-tokens N          Per-member token budget, nonzero (default 8)
  --prompt-0 TEXT         Member 0 prompt (default: a France question)
  --prompt-1 TEXT         Member 1 prompt (default: a planet question)
  --prompt-file-0 PATH    Read member 0's prompt from a file instead
  --prompt-file-1 PATH    Read member 1's prompt from a file instead

Peer parity: the handshake compares world_size, max_context,
prefill_chunk_tokens, max_draft_tokens, use_mtp, allow_cache_reuse and the
token EXACTLY, so both roles need the same --model, --context,
--tp-bootstrap-port, --tp-control-port, --tp-device, --tp-gid-index and
--tp-control-token. rank1 supplies prefill_chunk_tokens 512 from its prefill
policy and 0 draft tokens with MTP absent, and runs one runner with
max_pending_requests = 2 because gufo serve forces --max-pending 1 for TP=2.

Limits and hazards:
  * Wrap BOTH roles in an external timeout. The RDMA collective timeout is
    30 s, but the control response read waits up to 24 h.
  * The RDMA adapter maps fixed IOVA windows, so the two roles must run on
    separate hosts with no other RDMA process on either.
  * Each prompt must fit one prefill chunk (<= PrefillCapacity, which is
    min(2048, context)); a longer prompt is refused before any collective.
  * Start rank 0 first and rank 1 within 30 s of it: the RDMA bootstrap polls
    for 30 s on both sides, then gives up. rank 0 sends the cohort as soon as
    its OWN model is loaded, so rank 1 must be loaded before rank 0's first
    prefill collective, which also times out after 30 s. rank 1 logs
    "[c2-probe rank1] ready: ..." at that point.
  * This probe runs the dormant serial C2 plan. It proves the control plane,
    cohort admission, ordered member execution and cross-rank token agreement.
    It proves nothing about batched or faster C2 execution.

Exit reasons:
  0 pass   1 transport/model/execution failure   2 arguments
  3 reserved (the unimplemented rank-1 role was retired)
  4 envelope mismatch   5 digest mismatch   6 member order mismatch
  7 worker C2 error response
  8/9 member 0/1 exceeded its token budget
  10/11 member 0/1 tokens disagree with the local greedy output
  rank1 exits 0 when rank 0 closes the channel after the cohort and 1 on any
  worker failure; the C2 verdict is rank 0's exit code)");
}

/// Owns the single bound collective scope and releases it on every exit path.
/// `EndOperation` is purely local (verbs.cpp), so this never blocks.
class OperationScope {
 public:
  OperationScope(std::shared_ptr<q::rocm::Communicator> communicator,
                 std::uint64_t scope_id)
      : communicator_(std::move(communicator)), scope_id_(scope_id) {}

  ~OperationScope() {
    if (!bound_) {
      return;
    }
    try {
      std::string ignored;
      (void)communicator_->EndOperation(scope_id_, &ignored);
    } catch (...) {
    }
  }

  OperationScope(const OperationScope&) = delete;
  OperationScope& operator=(const OperationScope&) = delete;

  [[nodiscard]] bool Begin(std::string* error) {
    if (!communicator_->BeginOperation(scope_id_, error)) {
      return false;
    }
    bound_ = true;
    return true;
  }

  [[nodiscard]] bool End(std::string* error) {
    if (!bound_) {
      return true;
    }
    bound_ = false;
    return communicator_->EndOperation(scope_id_, error);
  }

 private:
  std::shared_ptr<q::rocm::Communicator> communicator_;
  const std::uint64_t scope_id_;
  bool bound_{false};
};

bool ParseUint(std::string_view text, std::uint32_t* value) {
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), *value);
  return error == std::errc{} && end == text.data() + text.size();
}

bool ParseUint64(std::string_view text, std::uint64_t* value) {
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), *value);
  return error == std::errc{} && end == text.data() + text.size();
}

bool ParsePort(std::string_view text, std::uint16_t* value) {
  std::uint32_t parsed = 0;
  if (!ParseUint(text, &parsed) || parsed == 0 || parsed > 65535) {
    return false;
  }
  *value = static_cast<std::uint16_t>(parsed);
  return true;
}

bool ReadPromptFile(const std::string& path, std::string* prompt) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return false;
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  if (!stream.good() && !stream.eof()) {
    return false;
  }
  *prompt = buffer.str();
  // Prompts are line oriented: a trailing newline is presentation, not a token.
  while (!prompt->empty() &&
         (prompt->back() == '\n' || prompt->back() == '\r')) {
    prompt->pop_back();
  }
  return true;
}

std::string DigestHex(const server::TpPlanDigest& digest) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string text;
  text.reserve(digest.size() * 2);
  for (const auto byte : digest) {
    text.push_back(kHex[byte >> 4]);
    text.push_back(kHex[byte & 0x0F]);
  }
  return text;
}

std::string TokensText(std::span<const std::int32_t> tokens) {
  std::string text;
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    if (index != 0) {
      text.push_back(' ');
    }
    text += std::to_string(tokens[index]);
  }
  return text;
}

/// The collective trace rank 0 issued for one member, kept for the report.
struct MemberTrace {
  std::vector<std::int32_t> tokens;
  std::size_t prefill_forwards{0};
  std::size_t decode_forwards{0};
  bool stopped_on_token{false};
  bool stopped_on_budget{false};
};

/// Index of the first position where two token runs disagree, or the shorter
/// length when one is a prefix of the other.
std::size_t FirstDifference(std::span<const std::int32_t> expected,
                            std::span<const std::int32_t> actual) {
  const std::size_t shared = std::min(expected.size(), actual.size());
  for (std::size_t index = 0; index < shared; ++index) {
    if (expected[index] != actual[index]) {
      return index;
    }
  }
  return shared;
}

/// Runs one member locally and records the exact collective trace rank 0
/// issued. Returns false only on a local execution failure, which is a
/// transport-level probe failure rather than a contract mismatch.
///
/// The body mirrors one worker member: `Sync` (one prefill forward), then the
/// `SelectNext`/`Advance` alternation with the length-limit short circuit.
bool RunMember(const std::shared_ptr<q::Model>& model,
               std::vector<std::int32_t> prompt, std::uint32_t budget,
               std::uint32_t context, MemberTrace* trace,
               std::string* error) {
  auto session = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      context, error);
  if (!session) {
    return false;
  }
  // `Sync` keeps the common prefix and feeds the rest, so a fresh session
  // prefills the whole prompt in ceil(P / PrefillCapacity) forwards. The
  // worker admits members one at a time at capacity 1, so its single resident
  // prefill is also unbounded and chunks identically.
  const std::size_t prefill_chunks =
      (prompt.size() + model->PrefillCapacity() - 1) / model->PrefillCapacity();
  if (!session->Sync(prompt, error)) {
    return false;
  }
  trace->prefill_forwards = prefill_chunks;

  const gufo::sampling::SamplingConfig sampling{
      .temperature = kGreedyTemperature,
  };
  gufo::sampling::SamplerState sampler(sampling);
  for (std::size_t produced = 0; produced < budget; ++produced) {
    const auto logits = session->Logits();
    if (logits.empty()) {
      *error = "session has no logits to sample from";
      return false;
    }
    // SelectNext: sample the current logits. No forward, no collective.
    const auto sampled =
        static_cast<std::int32_t>(sampler.Sample(logits));
    if (model->IsStopToken(sampled)) {
      trace->stopped_on_token = true;
      break;
    }
    trace->tokens.push_back(sampled);
    if (trace->tokens.size() >= budget) {
      // The distributed runner does not require a final token advance, so the
      // worker completes here without evaluating the last token.
      trace->stopped_on_budget = true;
      break;
    }
    // Advance: one Forward of a single row, i.e. num_layers collectives.
    if (!session->Evaluate(sampled, error)) {
      return false;
    }
    ++trace->decode_forwards;
  }
  return true;
}

/// Hosts the real rank-1 C2 worker and returns a probe exit reason.
///
/// `gufo serve` cannot be reused for this: it forces `--max-pending 1` for
/// TP=2, so `SubmitCohort` refuses every cohort at admission and answers a
/// silent C2 error without running a single collective. The probe therefore
/// builds `InferenceBackend` itself and supplies the policy the cohort needs.
/// Nothing here relaxes a production guard.
int RunRank1Worker(const std::string& model_path, std::uint32_t context,
                   std::uint32_t device, std::uint32_t gid,
                   const std::string& bootstrap_host,
                   std::uint16_t bootstrap_port, std::uint16_t control_port,
                   const std::string& control_token) {
  std::string error;

  // The bootstrap connect blocks for up to 30 s, so announce the target first:
  // an operator must be able to tell "connecting" from "hung".
  Say("rank1", "connecting to rank 0 at " + bootstrap_host + ":" +
                   std::to_string(bootstrap_port) + " (control port " +
                   std::to_string(control_port) + ")");

  // 1. RDMA FIRST, exactly like serve.cpp: the bootstrap connect and rank 0's
  //    bootstrap listen both block for 30 s, and the control connect needs the
  //    worker handshake that `load` performs, so any other order deadlocks.
  const q::rocm::IbrverbsConfig rdma{
      .rank = kRank1,
      .world_size = kWorldSize,
      // Rank 1 is the active side: the RDMA adapter calls Socket::Connect.
      .bootstrap_host = bootstrap_host,
      .bootstrap_port = bootstrap_port,
      .device_index = device,
      .gid_index = gid,
  };
  auto communicator = q::rocm::CreateIbrverbsCommunicator(rdma, &error);
  if (!communicator) {
    Warn("RDMA communicator failed: " + error);
    return kTransportFailure;
  }
  Say("rank1", "RDMA peer established with " + bootstrap_host +
                   " on bootstrap port " + std::to_string(bootstrap_port));

  // 2. Rank 1 connects; it never listens, so rank 0 must already be bound.
  auto control =
      server::TpControlChannel::Connect(bootstrap_host, control_port, &error);
  if (!control) {
    Warn("TP control connect failed: " + error);
    return kTransportFailure;
  }
  Say("rank1", "TP control connected to " + bootstrap_host + ":" +
                   std::to_string(control_port));

  // 3. `load` maps the model at rank 1, builds the runner pool and the
  //    scheduler, installs the communicator and performs the handshake
  //    itself, so the parity fields below come straight from the arguments.
  const server::TextPrefillPolicy prefill_policy{
      // Handshake `prefill_chunk_tokens`; it must equal rank 0's value.
      .decode_active_tokens = kWorkerPrefillChunkTokens,
  };
  const server::TextSchedulerPolicy scheduler_policy{
      // Cohort admission needs this; every other field keeps the production
      // default, so the scheduler behaves exactly as `gufo serve` does.
      .max_pending_requests = kWorkerMaxPendingRequests,
      // Rank 0 sends one member per client id, so each member is admitted on
      // its own and the per-client ceiling never blocks the cohort.
      .max_pending_requests_per_client = kWorkerMaxPendingRequestsPerClient,
  };
  // Defaults: backend kDisabled with an empty draft path, so the model loads
  // no MTP sidecar, the handshake reports 0 draft tokens and `use_mtp = false`.
  // The worker refuses a cohort while MTP is loaded (tp_cohort_worker.cpp), and
  // the C2 response contract cannot carry draft telemetry. `max_draft_tokens`
  // stays at its default 7 rather than 0: `Model::Load` rejects a zero draft
  // limit, and the handshake's draft field comes from `has_mtp`, not from it.
  const server::TextSpeculativeConfig speculative_config{};
  // Default: no directory, so the disk cache is disabled. TP=2 forbids one
  // (serve.cpp) and `BuildCohortMembers` requires uncached members.
  const server::TextDiskCacheConfig disk_cache_config{};
  const server::TextTpConfig tp_config{
      .rank = kRank1,
      .world_size = kWorldSize,
      .hip_device = static_cast<int>(device),
      .allow_cache_reuse = false,
      .communicator = communicator,
      .control = control,
      .auth_token = control_token,
  };
  server::InferenceBackend backend;
  if (!backend.load(model_path, &error,
                    // Handshake `max_context`; must equal rank 0's --context.
                    context, kWorkerSessionCount, prefill_policy,
                    scheduler_policy, speculative_config, disk_cache_config,
                    // No vision: `gufo serve` forbids it for TP=2 as well.
                    /*vision_model_path=*/"", tp_config)) {
    Warn("rank-1 worker load failed: " + error);
    return kTransportFailure;
  }
  Say("rank1",
      "ready: model loaded, TP control handshaken, runner pool capacity " +
          std::to_string(kWorkerSessionCount) + ", max_pending_requests " +
          std::to_string(kWorkerMaxPendingRequests) +
          ", no MTP; waiting for the C2 command");

  // 4. The worker loop blocks until rank 0 closes the channel or the worker
  //    fails; the C2 contract itself is only judged by rank 0.
  if (backend.run_worker(&error)) {
    Say("rank1", "worker loop returned without a failure");
    return kOk;
  }
  if (error.find(kPeerClosedChannel) != std::string::npos) {
    Say("rank1",
        "rank 0 closed the control channel; the C2 verdict is rank 0's "
        "exit code");
    return kOk;
  }
  Warn("rank-1 worker stopped: " + error);
  return kTransportFailure;
}

}  // namespace

int main(int argc, char** argv) {
  std::string role;
  std::string model_path = kDefaultModel;
  std::string prompt[kMemberCount] = {"The capital of France is",
                                      "The largest planet in the solar "
                                      "system is"};
  std::string prompt_file[kMemberCount];
  std::uint64_t cohort_id = 1;
  std::uint64_t sequence = 1;
  std::uint64_t member_id[kMemberCount] = {1, 2};
  std::uint32_t budget = 8;
  std::uint32_t context = 4096;
  std::uint32_t device = 0;
  std::uint32_t gid = 0;
  std::uint16_t bootstrap_port = 18515;
  std::uint16_t control_port = 18516;
  std::string bootstrap_host;
  std::string control_token;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : std::string();
    };
    if (arg == "--help" || arg == "-h") {
      Usage();
      return kOk;
    } else if (arg == "--role") {
      role = next();
    } else if (arg == "--model") {
      model_path = next();
    } else if (arg == "--cohort-id") {
      if (!ParseUint64(next(), &cohort_id)) {
        Warn("--cohort-id requires an unsigned integer");
        return kInvalidArguments;
      }
    } else if (arg == "--sequence") {
      if (!ParseUint64(next(), &sequence)) {
        Warn("--sequence requires an unsigned integer");
        return kInvalidArguments;
      }
    } else if (arg == "--member-id-0") {
      if (!ParseUint64(next(), &member_id[0])) {
        Warn("--member-id-0 requires an unsigned integer");
        return kInvalidArguments;
      }
    } else if (arg == "--member-id-1") {
      if (!ParseUint64(next(), &member_id[1])) {
        Warn("--member-id-1 requires an unsigned integer");
        return kInvalidArguments;
      }
    } else if (arg == "--max-tokens") {
      if (!ParseUint(next(), &budget)) {
        Warn("--max-tokens requires an unsigned integer");
        return kInvalidArguments;
      }
    } else if (arg == "--prompt-0") {
      prompt[0] = next();
    } else if (arg == "--prompt-1") {
      prompt[1] = next();
    } else if (arg == "--prompt-file-0") {
      prompt_file[0] = next();
    } else if (arg == "--prompt-file-1") {
      prompt_file[1] = next();
    } else if (arg == "--tp-bootstrap-port") {
      if (!ParsePort(next(), &bootstrap_port)) {
        Warn("--tp-bootstrap-port is out of range");
        return kInvalidArguments;
      }
    } else if (arg == "--tp-bootstrap-host") {
      bootstrap_host = next();
    } else if (arg == "--tp-control-port") {
      if (!ParsePort(next(), &control_port)) {
        Warn("--tp-control-port is out of range");
        return kInvalidArguments;
      }
    } else if (arg == "--tp-control-token") {
      control_token = next();
    } else if (arg == "--tp-device") {
      if (!ParseUint(next(), &device)) {
        Warn("--tp-device requires an unsigned integer");
        return kInvalidArguments;
      }
    } else if (arg == "--tp-gid-index") {
      if (!ParseUint(next(), &gid)) {
        Warn("--tp-gid-index requires an unsigned integer");
        return kInvalidArguments;
      }
    } else if (arg == "--context") {
      if (!ParseUint(next(), &context)) {
        Warn("--context requires an unsigned integer");
        return kInvalidArguments;
      }
    } else {
      Warn("unknown argument: " + arg);
      return kInvalidArguments;
    }
  }

  if (role.empty()) {
    Warn("--role rank0|rank1 is required; see --help");
    return kInvalidArguments;
  }
  if (role != "rank0" && role != "rank1") {
    Warn("--role must be rank0 or rank1");
    return kInvalidArguments;
  }
  // Only rank 0 owns prompts, so only rank 0 reads prompt files. Rank 1 has no
  // cohort to build and never issues a request of its own.
  if (role == "rank0") {
    for (std::size_t index = 0; index < kMemberCount; ++index) {
      if (!prompt_file[index].empty() &&
          !ReadPromptFile(prompt_file[index], &prompt[index])) {
        Warn("prompt file for member " + std::to_string(index) +
             " is unreadable: " + prompt_file[index]);
        return kInvalidArguments;
      }
    }
  }
  if (model_path.empty() || context == 0 || budget == 0) {
    Warn(
        "--model, a nonzero --context and a nonzero --max-tokens are "
        "required");
    return kInvalidArguments;
  }
  if (cohort_id == 0 || sequence == 0) {
    Warn("--cohort-id and --sequence must both be nonzero");
    return kInvalidArguments;
  }
  if (member_id[0] < kMinMemberId || member_id[1] < kMinMemberId ||
      member_id[0] == member_id[1]) {
    Warn("C2 member ids must be nonzero and distinct");
    return kInvalidArguments;
  }
  if (bootstrap_port == control_port) {
    Warn("--tp-bootstrap-port and --tp-control-port must differ");
    return kInvalidArguments;
  }
  if (control_token.empty() || control_token.size() > kMaxAuthTokenBytes) {
    Warn("--tp-control-token is required and must be at most 4096 bytes");
    return kInvalidArguments;
  }
  if (device >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      gid > std::numeric_limits<std::uint8_t>::max()) {
    Warn("--tp-device or --tp-gid-index is out of range");
    return kInvalidArguments;
  }
  // Rank 0 is the only role that builds a cohort, so only rank 0 rejects an
  // empty prompt.
  if (role == "rank0") {
    for (std::size_t index = 0; index < kMemberCount; ++index) {
      if (prompt[index].empty()) {
        Warn("member " + std::to_string(index) + " has an empty prompt");
        return kInvalidArguments;
      }
    }
  }

  // Rank 1 owns no cohort: it is rank 0 that builds, sends and validates the
  // command. Everything above is the shared transport and policy validation, so
  // a malformed rank-1 invocation fails here, before any RDMA or control peer
  // is touched.
  if (role == "rank1") {
    if (bootstrap_host.empty()) {
      Warn(
          "--role rank1 requires --tp-bootstrap-host with rank 0's address; "
          "the RDMA bootstrap and the TP control peer both connect there");
      return kInvalidArguments;
    }
    return RunRank1Worker(model_path, context, device, gid, bootstrap_host,
                          bootstrap_port, control_port, control_token);
  }

  std::string error;

  // 1. RDMA first: both bootstraps block for 30 s, and the control listener
  //    cannot accept before the worker connects, so the order is load-bearing.
  const q::rocm::IbrverbsConfig rdma{
      .rank = 0,
      .world_size = 2,
      // Empty bootstrap host makes rank 0 the AI_PASSIVE listener.
      .bootstrap_host = "",
      .bootstrap_port = bootstrap_port,
      .device_index = device,
      .gid_index = gid,
  };
  auto communicator = q::rocm::CreateIbrverbsCommunicator(rdma, &error);
  if (!communicator) {
    Warn("RDMA communicator failed: " + error);
    return kTransportFailure;
  }
  Say("rank0", "RDMA peer established on bootstrap port " +
                   std::to_string(bootstrap_port));

  // 2. The rank-zero control peer listens; the worker connects and both
  //    handshakes must agree field for field.
  auto control = server::TpControlChannel::Listen(control_port, &error);
  if (!control) {
    Warn("TP control listen failed: " + error);
    return kTransportFailure;
  }
  const server::TpControlConfig control_config{
      .rank = 0,
      .world_size = 2,
      .max_context = context,
      // Mandatory for C2: the worker refuses a cohort before binding a scope
      // while MTP is loaded, and the C2 response cannot carry draft
      // telemetry.
      .max_draft_tokens = 0,
      .use_mtp = false,
      .allow_cache_reuse = false,
      .auth_token = control_token,
      .prefill_chunk_tokens = kWorkerPrefillChunkTokens,
  };
  if (!control->Handshake(control_config, &error)) {
    Warn("TP control handshake failed: " + error);
    return kTransportFailure;
  }
  Say("rank0", "TP control handshaken on port " + std::to_string(control_port));

  // 3. Load with the communicator attached so the executor installs its
  //    all-reduce callback; a TP=2 model refuses to load without one.
  q::ModelOptions options{
      .max_context = context,
      // No MTP sidecar and no MTP path.
      .mtp_model_path = "",
      .max_draft_tokens = 7,
      .vision_model_path = "",
      // The C2 slice is serial: one resident request per rank.
      .decode_concurrency = 1,
      .tp_rank = 0,
      .tp_world_size = 2,
      .hip_device = static_cast<int>(device),
      .communicator = communicator,
  };
  auto model = q::Model::Load(model_path, options, &error);
  if (!model) {
    Warn("model load failed: " + error);
    return kTransportFailure;
  }
  if (model->HasMtp()) {
    Warn("model reports an MTP sidecar, which C2 refuses");
    return kInvalidArguments;
  }
  Say("rank0", "model loaded: layers=" +
                   std::to_string(model->config().num_layers) + " hidden=" +
                   std::to_string(model->config().hidden_size) +
                   " prefill_capacity=" +
                   std::to_string(model->PrefillCapacity()) + " context=" +
                   std::to_string(context));

  // 4. Build the two members and reject anything the worker would refuse
  //    before a single collective is issued.
  if (budget > context) {
    Warn("--max-tokens exceeds --context");
    return kInvalidArguments;
  }
  std::vector<std::int32_t> member_prompt[kMemberCount];
  for (std::size_t index = 0; index < kMemberCount; ++index) {
    member_prompt[index] = model->Tokenize(prompt[index]);
    if (member_prompt[index].empty()) {
      Warn("member " + std::to_string(index) + " prompt tokenized empty");
      return kInvalidArguments;
    }
    if (member_prompt[index].size() > model->PrefillCapacity()) {
      Warn("member " + std::to_string(index) + " prompt has " +
           std::to_string(member_prompt[index].size()) +
           " tokens and needs more than one prefill chunk (capacity " +
           std::to_string(model->PrefillCapacity()) +
           "); a multi-chunk prefill is not mirrored by this probe");
      return kInvalidArguments;
    }
    if (member_prompt[index].size() + budget >
        static_cast<std::size_t>(context)) {
      Warn("member " + std::to_string(index) +
           " prompt plus budget exceeds the negotiated context");
      return kInvalidArguments;
    }
  }

  server::TpControlCommand command;
  // Every legacy C1 field stays default: `HasLegacyCommandFields` hard-rejects
  // a C2 command that still carries one of them.
  command.sequence = sequence;
  command.kind = server::TpControlCommandKind::kCohort2Ar;
  command.cohort_id = cohort_id;
  for (std::size_t index = 0; index < kMemberCount; ++index) {
    // Distinct, nonempty client ids: `FitsClientLimits` counts pending
    // requests per client and the worker admits one per client.
    server::TpControlMemberRequest member;
    member.member_id = member_id[index];
    member.max_tokens = budget;
    member.cache_prompt = false;
    member.cache_prefix_tokens = 0;
    member.prompt_tokens = member_prompt[index];
    member.client_id = "tp-c2-probe-member-" + std::to_string(index);
    command.members.push_back(std::move(member));
  }
  // Digests come from the exported canonical functions, never by hand.
  command.execution_plan_digest =
      server::ComputeTpExecutionPlanDigest(command);
  command.cache_plan_digest = server::ComputeTpCachePlanDigest(command);
  if (!server::ValidateTpControlCommand(command, &error)) {
    Warn("local C2 command pre-flight failed: " + error);
    return kInvalidArguments;
  }
  Say("rank0", "C2 command sequence=" + std::to_string(command.sequence) +
                   " cohort=" + std::to_string(command.cohort_id) +
                   " execution_plan=" +
                   DigestHex(command.execution_plan_digest) +
                   " cache_plan=" + DigestHex(command.cache_plan_digest));

  // 5. `BeginOperation` is local, so it needs no ordering against the worker's
  //    own bind. The scope stays bound across both members: the C2 members
  //    share the command sequence scope, so one lease spans the whole cohort.
  OperationScope operation(communicator, command.sequence);
  if (!operation.Begin(&error)) {
    Warn("collective scope bind failed: " + error);
    return kTransportFailure;
  }
  if (!control->SendCommand(command, &error)) {
    Warn("C2 command send failed: " + error);
    return kTransportFailure;
  }
  Say("rank0", "C2 command sent; scope " + std::to_string(command.sequence) +
                   " bound for the whole cohort");

  // 6. Lockstep: issue exactly the forwards the worker issues, in order.
  MemberTrace trace[kMemberCount];
  for (std::size_t index = 0; index < kMemberCount; ++index) {
    if (!RunMember(model, member_prompt[index], budget, context, &trace[index],
                   &error)) {
      Warn("member " + std::to_string(index) + " local execution failed: " +
           error);
      return kTransportFailure;
    }
    const std::size_t forwards =
        trace[index].prefill_forwards + trace[index].decode_forwards;
    const char* stop = trace[index].stopped_on_token
                           ? "token"
                           : (trace[index].stopped_on_budget ? "budget"
                                                             : "none");
    Say("rank0", "member " + std::to_string(index) +
                     " prompt=" + std::to_string(member_prompt[index].size()) +
                     "t tokens=" + std::to_string(trace[index].tokens.size()) +
                     " prefill_forwards=" +
                     std::to_string(trace[index].prefill_forwards) +
                     " decode_forwards=" +
                     std::to_string(trace[index].decode_forwards) +
                     " forwards=" + std::to_string(forwards) +
                     " collectives=" +
                     std::to_string(forwards * model->config().num_layers) +
                     " stop=" + stop + " text=" +
                     model->Decode(trace[index].tokens));
  }

  // 7. Release the scope before answering, exactly like the worker tail.
  if (!operation.End(&error)) {
    Warn("collective scope release failed: " + error);
    return kTransportFailure;
  }
  server::TpControlResponse response;
  if (!control->ReceiveResponse(&response, &error)) {
    Warn("C2 response receive failed: " + error);
    return kTransportFailure;
  }

  // 8. Verify the response envelope. `ReceiveResponse` already re-validated
  //    the frame, but nothing checks that it answers THIS command.
  if (response.sequence != command.sequence) {
    Warn("response sequence " + std::to_string(response.sequence) +
         " does not match the command sequence " +
         std::to_string(command.sequence));
    return kEnvelopeMismatch;
  }
  if (response.kind != server::TpControlResponseKind::kCohort2Ar) {
    Warn("response kind " +
         std::to_string(static_cast<std::uint32_t>(response.kind)) +
         " is not kCohort2Ar");
    return kEnvelopeMismatch;
  }
  if (response.cohort_id != command.cohort_id) {
    Warn("response cohort " + std::to_string(response.cohort_id) +
         " does not match the command cohort " +
         std::to_string(command.cohort_id));
    return kEnvelopeMismatch;
  }
  // The library does NOT compare the echoed digests; that is the broker's job.
  if (response.execution_plan_digest != command.execution_plan_digest ||
      response.cache_plan_digest != command.cache_plan_digest) {
    Warn("response plan digests do not match the command: response=" +
         DigestHex(response.execution_plan_digest) + "/" +
         DigestHex(response.cache_plan_digest) + " command=" +
         DigestHex(command.execution_plan_digest) + "/" +
         DigestHex(command.cache_plan_digest));
    return kDigestMismatch;
  }
  if (response.members.size() != kMemberCount) {
    Warn("response carries " + std::to_string(response.members.size()) +
         " members, expected " + std::to_string(kMemberCount));
    return kMemberOrderMismatch;
  }
  for (std::size_t index = 0; index < kMemberCount; ++index) {
    if (response.members[index].member_id != command.members[index].member_id) {
      Warn("response member " + std::to_string(index) + " has id " +
           std::to_string(response.members[index].member_id) +
           ", expected " + std::to_string(command.members[index].member_id) +
           " (member order is part of the C2 contract)");
      return kMemberOrderMismatch;
    }
  }
  // A C2 error response is a cohort refusal, not a token disagreement.
  if (!response.error.empty()) {
    std::fprintf(stderr,
                 "[c2-probe] C2 ERROR RESPONSE for cohort %llu: %s\n",
                 static_cast<unsigned long long>(response.cohort_id),
                 response.error.c_str());
    std::fflush(stderr);
    return kCohortErrorResponse;
  }

  // 9. Per-member budget and cross-rank token agreement. The reduction is a
  //    host float add over two rank-local partials, so bitwise logit equality
  //    is not claimed; the contract under test is the token sequence.
  const int budget_exceeded[kMemberCount] = {kMember0BudgetExceeded,
                                            kMember1BudgetExceeded};
  const int token_mismatch[kMemberCount] = {kMember0TokenMismatch,
                                           kMember1TokenMismatch};
  for (std::size_t index = 0; index < kMemberCount; ++index) {
    const auto& received = response.members[index].tokens;
    if (received.size() > budget) {
      Warn("member " + std::to_string(index) + " returned " +
           std::to_string(received.size()) + " tokens, budget is " +
           std::to_string(budget));
      return budget_exceeded[index];
    }
  }
  for (std::size_t index = 0; index < kMemberCount; ++index) {
    const auto& received = response.members[index].tokens;
    const std::span<const std::int32_t> local(trace[index].tokens);
    if (std::ranges::equal(local, received)) {
      Say("rank0", "member " + std::to_string(index) +
                       " AGREES with the local greedy output: " +
                       TokensText(received));
      continue;
    }
    // `at` is the shorter length when one run is a prefix of the other, so a
    // token may be missing on either side; report it as end-of-run instead of
    // indexing past the end.
    const std::size_t at = FirstDifference(local, received);
    const std::string worker_token =
        at < received.size() ? std::to_string(received[at]) : "end-of-run";
    const std::string local_token =
        at < local.size() ? std::to_string(local[at]) : "end-of-run";
    Warn("member " + std::to_string(index) +
         " token disagreement at index " + std::to_string(at) +
         ": worker=" + worker_token + " local=" + local_token +
         " worker_len=" + std::to_string(received.size()) +
         " local_len=" + std::to_string(local.size()) +
         " worker_tokens=[" + TokensText(received) + "] local_tokens=[" +
         TokensText(local) + "]");
    return token_mismatch[index];
  }

  Say("rank0",
      "C2 cohort PASS: control plane, ordered two-member execution and "
      "cross-rank token agreement verified");
  return kOk;
}

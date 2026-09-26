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
//   exactly `num_layers` `ExchangePartial` collectives of `n * hidden_size * 4`
//   bytes (executor.cpp: one `Moe` -> one `AllReduce` per layer). Byte size is
//   therefore fully determined by the forward's row count.
//
//   The worker runs each cohort member as independent serial C1 work, so for
//   member m with prompt P, budget K and prefill capacity
//   C = `PrefillCapacity()` (min(2048, context)), without MTP:
//     prefill: ceil(P / C) forwards of n = min(C, remaining) rows each, in
//              chunk order. Rank 0's `Session::Sync(P)` chunks inside `Feed`
//              at `exec.max_batch`, which `Model::Load` sets to
//              `PrefillCapacity()` (engine.cpp:190, 599-620). The worker's
//              Qwen Flash-Next `Prefill` caps `consumed` at the same
//              `PrefillCapacity` and re-`Sync`s the CUMULATIVE prefix, whose
//              common-prefix skip feeds exactly the new tail
//              (inference_backend.cpp:2646-2672), so it issues the same
//              forwards with the same row counts in the same order. The member
//              is the only resident request (runner pool capacity 1) and no
//              decoder is runnable, so `bounded_prefill` is false
//              (text_generation_scheduler.cpp:568-572): the worker's per-step
//              budget is the whole remaining prompt, and it is the
//              `PrefillCapacity` cap, NOT the handshake's
//              `prefill_chunk_tokens`, that chunks a long prompt.
//              Nothing else in the worker bounds a prompt: `BuildCohortMembers`
//              only requires an uncached member, and the runner's own length
//              check is `max_context` (text_model_runner.cpp:1273).
//     decode:  the scheduler alternates `SelectNext()` (samples the current
//              logits, NO forward) with `Advance()` (= `Evaluate` = 1 Forward
//              of one row). `TextRunnerCapabilities::final_token_advance_-
//              required` is false for the distributed Flash-Next runner, so
//              `PrepareDecode` completes the request at the token limit
//              WITHOUT advancing the final token.
//     => tokens published T, decode forwards A = T - 1 when the run ends on
//        the length limit, and A = T when `SelectNext` returns a stop token
//        first. Per member: (ceil(P / C) + A) * num_layers collectives, whose
//        prefill byte sequence is `num_layers` copies of
//        `min(C, remaining) * hidden_size * 4` per chunk, in chunk order, and
//        whose decode byte sequence is A * num_layers copies of
//        `hidden_size * 4`.
//
//   `RunMember` below reproduces exactly that: one `session->Sync(prompt)`,
//   then sample from `session->Logits()` (no collective), stop on
//   `model->IsStopToken`, stop at the budget, and only then
//   `session->Evaluate(token)` (one collective group). It deliberately
//   does NOT use `Session::DecodeStep`, which always advances the token it
//   samples and would therefore issue one extra forward per member and
//   desynchronise the trace. A FRESH session per member is what makes the
//   prefill start from an empty prefix, which is also what the worker does for
//   an uncached member (`cache_prompt = false`, `allow_cache_reuse = false`),
//   so `Sync`'s common-prefix skip is zero on both sides and both chunk the
//   whole prompt.
//
//   The probe OBSERVES the prefill byte sequence rather than trusting the
//   formula: rank 0 installs a `CollectiveTrace` decorator on the model
//   communicator, so the trace carries the byte size of every collective rank 0
//   actually issued, and `PlanPrefill`'s prediction is compared against it
//   before the member line is reported. A mismatch is exit 13, never a silent
//   pass: a wrong prefill chunk count is the one assumption that would
//   otherwise desynchronise both ranks into an unattributable 30 s collective
//   timeout, and the log names the first divergent index and both byte sizes.
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
//     `InferenceBackend` in-process with `--worker-max-pending` (default 2),
//     runner pool capacity 1, and `--context` equal to rank 0's. It loads no
//     MTP sidecar unless `--mtp-model` injects one; see the third fault mode.
//     The serving front door's own TP=2 C1 guard is deliberately NOT
//     reproduced here; see the admission-refusal fault mode below.
//   * Rank 1 never validates the C2 contract; it reports what its worker loop
//     returns. The C2 verdict is always rank 0's exit code.
//   * Every `[c2-probe]` line is stamped with `t=<ms>ms` elapsed since process
//     start, because a 30 s collective timeout and a 30 s model load are
//     indistinguishable in an unstamped log.
//
// FAULT INJECTION (deliberate; every mode fails closed)
//   The fail-closed C2 paths are otherwise covered only by hosted tests with
//   fakes. A real RDMA failure does not arrive as "lease Begin() returned
//   false"; it arrives as a 30 s `kCollectiveTimeout` or a header mismatch
//   inside `ExchangePartial`, which sets `poisoned_` on the communicator
//   (verbs.cpp:604-608) and surfaces through a different route. These three
//   modes put that route in front of a real worker over a real RDMA pair.
//
//   1. ADMISSION REFUSAL (rank 1 `--worker-max-pending 1`)
//      `SubmitCohort` requires `queued_count + 2 <= max_pending_requests`
//      (text_generation_scheduler.cpp:1515-1519), so 1 refuses the cohort
//      before either member reaches a runner. The C2 seam catches the submit
//      exception, releases the operation lease and sends a wire-valid C2 error
//      response that still carries BOTH member envelopes
//      (tp_cohort_plan.cpp `MakeCohortEnvelope`). No collective is issued, so
//      no RDMA traffic and no timeout: the run is fast and deterministic, and
//      rank 0 exits with the C2-error reason (7) after validating the envelope,
//      the digests and the member order of a response it did not ask for to
//      succeed. Rank 0 MUST also be given `--expect-c2-error`, because the
//      lockstep driver would otherwise issue a prefill collective that nobody
//      answers and wait out the 30 s collective timeout.
//
//      The `gufo serve` guard that rejects `max_pending == 1` for TP=2
//      (serve.cpp:1210-1219) is NOT replicated. The probe deliberately bypasses
//      the serving front door and hosts `InferenceBackend` in-process, and
//      forcing the refusal is the entire point of this mode. Nothing outside
//      this test binary changes and no production guard is relaxed.
//
//   2. OPERATION-SCOPE MISMATCH (rank 0 `--tp-scope-override N`)
//      Rank 0 binds `communicator->BeginOperation(N)` while still sending a
//      command whose `sequence` is the normal value, so rank 1's worker binds
//      `command.sequence` (inference_backend.cpp `TpOperationScopeLease`,
//      tp_cohort_worker.cpp) and the two scopes disagree. The FIRST
//      `ExchangePartial` header exchange fails `magic`/`version`/`scope_id`/
//      `operation_id`/`bytes` validation (verbs.cpp:644-660) and poisons both
//      communicators, so the run fails closed on both ranks: rank 1's model
//      execution errors, the member `Wait()` throws, the seam cancels the peer
//      and the release itself fails on a poisoned communicator, so the worker
//      stops. Rank 0 reports a collective/poison error and exits 1, never a
//      token mismatch and never a PASS. There is deliberately NO flag that
//      makes this mode look like a success, and `N` must differ from
//      `--sequence` so the override can never be a silent no-op.
//
//   3. MTP-SIDECAR REFUSAL (rank 1 `--mtp-model PATH`, rank 0
//      `--expect-worker-mtp` with `--expect-c2-error`)
//      `RunTpCohortCommand` refuses a cohort while MTP is loaded BEFORE it
//      binds the operation scope (tp_cohort_worker.cpp:164-166, `kMtpRefusal`),
//      so on this path the lease bind and the first collective are dead code.
//      `Impl::State::tp_use_mtp` comes from `Model::HasMtp`
//      (inference_backend.cpp:3839) and the handshake then presents
//      `use_mtp = true` with `max_draft_tokens = N` (inference_backend.cpp:
//      3841-3846), which rank 0 must mirror field for field or `Handshake`
//      rejects the pair (tp_control.cpp:671-680).
//      Rank 0 loads NO sidecar: it is the AR lockstep driver, the point of
//      the mode is that no collective is issued anywhere, and
//      `--expect-c2-error` already removes rank 0's own lockstep. It declares
//      the worker's parity with `--expect-worker-mtp` plus the same
//      `--draft-tokens` width and loads exactly the AR model of the PASS path.
//      EXPECTED: rank 1 loads the sidecar, handshakes, logs its ready line
//      with the draft width and refuses the cohort. The refusal is decided
//      before `BeginOperation`, so rank 1 binds NO lease and issues NO
//      collective, and the wire-valid C2 error response (which still carries
//      BOTH member envelopes, `BuildCohortFailureResponse`) answers within
//      single-digit milliseconds of the command. Rank 0 finds `error` exactly
//      "C2 cohort AR is not supported while the MTP sidecar is loaded" and
//      exits 7, having logged no `prefill complete` and no `member N` line
//      because the lockstep never ran. Rank 1 keeps serving and exits 0 when
//      rank 0 closes the channel. The MTP refusal precedes member translation
//      and admission, so `--worker-max-pending 1` combined with `--mtp-model`
//      still ends in this refusal rather than in mode 1, and
//      `--tp-scope-override` is inert here because no collective is issued for
//      it to poison.
//
// EXIT REASONS
//   0  every C2 contract and token-agreement check passed (rank 0), or the
//      control channel closed after the cohort (rank 1)
//   1  transport/model/execution failure (RDMA, control, decode); this is the
//      `--tp-scope-override` verdict
//   2  argument or pre-flight validation failure
//   3  reserved; retired with the unimplemented rank-1 stub
//   4  response envelope mismatch (sequence, kind or cohort id)
//   5  response plan-digest mismatch
//   6  response member count or member order mismatch
//   7  the worker answered with a C2 error response; this is the
//      `--worker-max-pending 1`, `--mtp-model` and `--expect-c2-error` verdict
//   8  member 0 returned more tokens than its budget
//   9  member 1 returned more tokens than its budget
//  10  member 0 tokens disagree with rank 0's local greedy output
//  11  member 1 tokens disagree with rank 0's local greedy output
//  12  a C2 error response arrived, but not the MTP sidecar refusal that
//      `--expect-worker-mtp` asserts; the run still failed closed, but it did
//      not fail for the injected reason
//  13  the collective byte sequence rank 0 ISSUED for a member's prefill
//      differs from the sequence `PlanPrefill` PREDICTED from the chunking;
//      the executed and reported prefill forward counts disagree, so the
//      lockstep would not mirror the worker
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
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
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
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
/// `--worker-max-pending` overrides it so the refusal can be provoked.
constexpr std::uint32_t kWorkerMaxPendingRequests = 2;
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
/// First id `--prompt-tokens-N` emits. The band is deliberately LOW: the head
/// of a byte-level BPE vocabulary is ordinary text bytes, so a synthetic prompt
/// stays well conditioned instead of probing arbitrary embedding rows, and the
/// walk starts at 1 rather than 0.
constexpr std::int32_t kSyntheticFirstToken = 1;
/// `--prompt-tokens-N` walks the low band with this stride. It is coprime with
/// `kSyntheticSpan` (512) so the sequence visits every id in the band before it
/// repeats, which keeps a long synthetic prompt from degenerating into a
/// trivially periodic one.
constexpr std::size_t kSyntheticStride = 5;
/// Width of the low band `--prompt-tokens-N` walks. Clamped to `vocab - 1` at
/// run time so the emitted ids are always inside the embedding.
constexpr std::size_t kSyntheticSpan = 512;
/// Member m's synthetic prompt starts this far into the walk, so the two
/// members carry DIFFERENT prompts of equal length: the C2 digests stay
/// distinguishable and a mixed cohort (one short member, one long member) is
/// expressible without a second mechanism.
constexpr std::size_t kSyntheticMemberOffset = 11;
/// `run_worker` reports rank 0's exit as a closed channel (tp_control.cpp
/// `RecvAll`). That is the one worker stop that is not a rank-1 failure.
constexpr std::string_view kPeerClosedChannel =
    "TP control peer closed the channel";
/// Verbatim copy of the worker's MTP refusal reason (tp_cohort_worker.cpp
/// `kMtpRefusal`). The third fault mode compares the C2 error against it
/// EXACTLY, because any refusal answers with the same exit 7: without the
/// comparison an admission refusal could be read as this mode's verdict.
constexpr std::string_view kMtpRefusal =
    "C2 cohort AR is not supported while the MTP sidecar is loaded";
/// Monotonic origin for every `[c2-probe] t=<ms>ms` stamp. `steady_clock` never
/// steps, so a difference between two stamps is a real elapsed duration rather
/// than a wall-clock jump.
const std::chrono::steady_clock::time_point kProbeStart =
    std::chrono::steady_clock::now();

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
  /// A C2 error response arrived, but not the refusal `--expect-worker-mtp`
  /// asserts. Every refusal shares exit 7, so without this the third mode could
  /// not tell its own verdict from any other refusal.
  kC2ErrorReasonMismatch = 12,
  /// The prefill byte sequence rank 0 actually ISSUED for a member differs from
  /// the sequence `PlanPrefill` PREDICTED from the chunking. The two sides
  /// would not mirror each other, and a driver that reported the prediction as
  /// if it were the observation would turn that desync into a silent pass, so
  /// it is its own hard failure rather than a transport error or a logged
  /// warning.
  kCollectiveTraceDesync = 13,
};

/// Milliseconds elapsed since process start.
///
/// The RDMA collective timeout is 30 s and a model load is also tens of
/// seconds, so an unstamped lifecycle log cannot attribute a stall to either.
/// Every lifecycle line therefore carries this stamp, and the absence of a gap
/// between two stamps is itself the evidence that a phase was fast.
std::string ElapsedMs() {
  return std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - kProbeStart)
                            .count());
}

void Say(const char* role, const std::string& line) {
  const std::string stamp = ElapsedMs();
  std::printf("[c2-probe %s t=%sms] %s\n", role, stamp.c_str(), line.c_str());
  std::fflush(stdout);
}

void Warn(const std::string& line) {
  const std::string stamp = ElapsedMs();
  std::fprintf(stderr, "[c2-probe t=%sms] %s\n", stamp.c_str(), line.c_str());
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

Step-2 spike:
  --batched-w2            Both ranks run the SAME width-2 batched program over
                          the engine: two live sessions, per-member prefill
                          (prefill has no batch form), then one two-row
                          EvaluateBatch per decode step. No cohort, no command,
                          no control wire and no worker, so the only variable is
                          whether a batched forward keeps both ranks on one
                          collective schedule. Each rank prints its per-member
                          tokens, a checksum, and the exact member set included
                          in every batched step; diff the two logs to check both
                          token agreement and schedule agreement. The scope id
                          is a constant both sides bind by construction, so a
                          run is only comparable when BOTH ranks pass this flag.
  --serial-w2             The serial baseline for --batched-w2: the same
                          program, but each decode step advances the included
                          members one at a time. Both modes print decode_ms and
                          ms_per_step, so the pair measures what batching buys.
                          A member that stops early leaves a serial step with
                          one forward: compare ms_per_forward across runs whose
                          members stop at different steps.
                          BOTH ranks must pass the same mode flag.
  --allreduce-bench N     Load no model: time N all-reduces each of 1, 2 and 8
                          decode rows and one 512-row prefill chunk, and print
                          min/p50/mean/p99 microseconds per collective. BOTH
                          ranks must pass the same N.

Rank-one options:
  --tp-bootstrap-host HOST  REQUIRED for rank1. Rank 0's address, used for the
                          RDMA bootstrap and for the TP control connect. rank0
                          ignores it and binds the bootstrap to all interfaces.
  --worker-max-pending N   TextSchedulerPolicy max_pending_requests for the
                          rank-1 worker, >= 1 (default 2). The per-client
                          ceiling stays at 1 either way, and the runner pool
                          capacity stays at 1, so this moves nothing but the
                          admission ceiling. rank0 ignores it.
  --mtp-model PATH        Fault injection: load this real MTP draft sidecar
                          before the handshake, so the worker's state reports
                          a draft sidecar and the C2 worker refuses the cohort.
                          The path must be a readable file. Empty by default,
                          which is the no-MTP production policy. Production
                          path: /opt/models/qwen3.8-flash-next/MTP/
                          mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf. rank0
                          REFUSES this flag: it drives the AR lockstep and
                          must load no sidecar.

Peer options (both roles; the values must match or the handshake rejects):
  --model PATH            First GGUF shard (default: production Q4 path)
  --tp-bootstrap-port N   RDMA bootstrap port (default 18515)
  --tp-control-port N     TP control port (default 18516)
  --tp-control-token TOK  Nonempty token; must equal the peer's, <= 4096 B
  --tp-device N           HIP device index (default 0)
  --tp-gid-index N        RDMA GID index (default 0)
  --context N             Handshake context; must equal the peer's
                          (default 4096)
  --draft-tokens N        MTP draft width, 1..7 (default 7). rank1 loads it
                          with --mtp-model; rank0 declares the worker's
                          handshake max_draft_tokens with --expect-worker-mtp.
                          Without one of those two flags it injects no fault
                          and is refused.
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
  --prompt-tokens-0 N     Synthesize member 0's prompt as exactly N token ids
                          instead of tokenizing text. Refused together with
                          --prompt-0 or --prompt-file-0 for the same member.
  --prompt-tokens-1 N     The same for member 1.
                          This is how a prompt longer than one prefill chunk is
                          exercised: a synthetic sequence has an EXACT token
                          count, so the chunking, the plan digest and the
                          collective byte sequence are all reproducible, which
                          prose cannot promise (one edit of the vocabulary can
                          change a text prompt's token count). Ids are walked
                          from 1 with stride 5 across the first 512 ids of the
                          vocabulary, member 1 starting 11 steps in, skipping
                          any id the model reports as a stop token. Set only one
                          member for a mixed cohort: one single-chunk and one
                          multi-chunk member inside the same collective scope.
  --tp-scope-override N   Fault injection: bind collective scope N instead of
                          --sequence, while still sending a command whose
                          sequence is --sequence. N must be nonzero and must
                          differ from --sequence. rank1 ignores it.
  --expect-c2-error      Fault injection: skip the lockstep member execution and
                          wait only for the worker's response. rank1 ignores it.
  --expect-worker-mtp     Fault injection: declare that rank 1's handshake
                          presents a draft sidecar, so rank 0 mirrors
                          use_mtp=true and the worker's --draft-tokens. Rank 0
                          still loads NO sidecar and issues no collective; it
                          requires --expect-c2-error. rank1 ignores it.

Fault-injection modes (all deliberate, all fail closed; none can pass):
  1. Admission refusal.
       rank1: --worker-max-pending 1
       rank0: --expect-c2-error
     SubmitCohort requires queued_count + 2 <= max_pending_requests, so 1
     refuses the cohort at admission before either member reaches a runner. The
     seam catches the submit exception, releases the operation lease and sends
     a wire-valid C2 error response that still carries BOTH member envelopes.
     No collective is ever issued, so no RDMA traffic and no timeout.
     EXPECTED: fast and deterministic; rank 0 validates the envelope, the plan
     digests and the member order, finds a nonempty error and exits 7; rank 1
     keeps serving, sees rank 0 close the channel and exits 0.
     rank 0 needs --expect-c2-error because its lockstep driver would
     otherwise issue a prefill collective that nobody answers and wait out the
     30 s collective timeout. The C2 verdict stays rank 0's exit code, so
     rank 1 must also be started with --worker-max-pending 1.
  2. Operation-scope mismatch.
       rank0: --tp-scope-override N  (N != --sequence)
     Rank 0 binds communicator->BeginOperation(N) while the command still
     carries the normal sequence, so rank 1's worker binds command.sequence
     and the two scopes disagree. The FIRST ExchangePartial header exchange
     fails identity validation and poisons both communicators.
     EXPECTED: the run FAILS on both ranks. Rank 1's model execution errors,
     the member Wait() throws, the seam cancels the peer and the lease release
     itself fails on a poisoned communicator, so the worker stops. Rank 0
     reports a collective/poison error and exits 1, immediately after its first
     prefill collective. It must never report a token mismatch or a PASS, and
     there is deliberately no flag that makes this mode look like a success.
  3. MTP sidecar refusal.
       rank1: --mtp-model PATH [--draft-tokens N]
       rank0: --expect-worker-mtp [--draft-tokens N] --expect-c2-error
     The worker refuses any C2 cohort while an MTP draft sidecar is loaded, and
     it decides that BEFORE BeginOperation (tp_cohort_worker.cpp:164-166,
     kMtpRefusal), so no lease is bound and no collective is issued.
     EXPECTED: rank 1 loads the sidecar, handshakes, logs its ready line with
     the draft width and refuses. The C2 error response still carries BOTH
     member envelopes and arrives within single-digit milliseconds of the
     command. Rank 0 validates the envelope, the plan digests and the member
     order, finds error exactly "C2 cohort AR is not supported while the MTP
     sidecar is loaded" and exits 7, with no "prefill complete" and no
     "member N" line because the lockstep never ran; rank 1 keeps serving,
     sees rank 0 close the channel and exits 0.
     Rank 0 declares the worker's parity instead of loading a sidecar, because
     the point of the mode is that NO collective is issued anywhere. Rank 1
     must therefore be started with --mtp-model and rank 0 with
     --expect-worker-mtp; a disagreement is a handshake rejection, not a
     verdict. A --worker-max-pending 1 on rank 1 is never reached, because the
     MTP refusal precedes admission, and --tp-scope-override is inert on rank
     0 here because the skipped lockstep issues no collective for it to poison.

Peer parity: the handshake compares world_size, max_context,
prefill_chunk_tokens, max_draft_tokens, use_mtp, allow_cache_reuse and the
token EXACTLY, so both roles need the same --model, --context,
--tp-bootstrap-port, --tp-control-port, --tp-device, --tp-gid-index and
--tp-control-token. rank1 supplies prefill_chunk_tokens 512 from its prefill
policy and 0 draft tokens with MTP absent, and runs one runner with
max_pending_requests = --worker-max-pending (default 2) because gufo serve
forces --max-pending 1 for TP=2. That serving guard is deliberately NOT
reproduced here: the probe bypasses the serving front door, so
--worker-max-pending 1 reaches the scheduler and provokes the refusal instead
of being rejected.

Handshake parity, all eight fields compared (tp_control.cpp:671-680):
  field                 pass and modes 1-2         mode 3
  rank                  0 / 1                       0 / 1
  world_size            2 / 2                       2 / 2
  max_context           --context / --context       --context / --context
  prefill_chunk_tokens  512 / 512                    512 / 512
  max_draft_tokens      0 / 0                        --draft-tokens, both
                                                    roles, 7 when unset
  use_mtp               false / false                true / true
  allow_cache_reuse     false / false                false / false
  auth_token            --tp-control-token, both     --tp-control-token,
                                                    both

Limits and hazards:
  * Wrap BOTH roles in an external timeout. The RDMA collective timeout is
    30 s, but the control response read waits up to 24 h.
  * Every [c2-probe] line is stamped t=<ms>ms from process start, so a 30 s
    collective timeout, a 30 s model load and a 30 s bootstrap poll are
    distinguishable. Read the gaps, not the line count.
  * The listener is NOT retryable: rank 0's bootstrap Listen polls once for
    30 s and then fails, while rank 1's Connect retries until its own 30 s
    deadline. Start rank 0 FIRST and start rank 1 within that window, or the
    rendezvous cannot form.
  * Both containers need `--network host`. Without it each role gets its own
    network namespace and rank 0's 0.0.0.0 bootstrap listener is unreachable,
    which surfaces only as a 30 s accept timeout on rank 0 and an
    "Operation now in progress" connect failure on rank 1.
  * The RDMA adapter maps fixed IOVA windows, so the two roles must run on
    separate hosts with no other RDMA process on either.
  * A prompt MAY span several prefill chunks. It is prefilled in
    ceil(tokens / PrefillCapacity) forwards of min(capacity, remaining) rows,
    and PrefillCapacity is min(2048, --context), so a prompt longer than
    min(2048, context) is the case this exercises. Use --prompt-tokens-0/-1
    for it: the token count must be exact for the plan digest and the chunk
    arithmetic to be reproducible, and a text prompt's count is only known
    after the model is loaded.
  * Prompt + budget must fit the negotiated --context. For a --prompt-tokens
    value that is checked at t=0 ms, before any RDMA peer exists; for a text or
    file prompt it can only be checked after the model is loaded, because the
    token count is not known until then.
  * Start rank 0 first and rank 1 within 30 s of it: the RDMA bootstrap polls
    for 30 s on both sides, then gives up. rank 0 sends the cohort as soon as
    its OWN model is loaded, so rank 1 must be loaded before rank 0's first
    prefill collective, which also times out after 30 s. rank 1 logs
    "[c2-probe rank1] t=<ms>ms ready: ..." at that point.
  * This probe runs the dormant serial C2 plan. It proves the control plane,
    cohort admission, ordered member execution and cross-rank token agreement.
    It proves nothing about batched or faster C2 execution.
  * The three fault-injection modes above are the only supported ways to make a
    run fail. None relaxes a production guard, and none can report a PASS:
    --expect-c2-error only removes rank 0's collectives, and a response
    without an error in that mode is itself an envelope mismatch (exit 4).

Reading a long-prompt run (the collective byte sequence is CHUNKED):
  Rank 0 prints, per member, the prefill it PREDICTED and then the prefill it
  ISSUED. A run is only a PASS when the two agree; a divergence is exit 13 and
  the log names the first index at which they differ. With L the layer count
  and H the hidden size the "model loaded" line reports, C the capacity and
  r0..rk-1 the chunk row counts:
    [c2-probe rank0 t=Nms] member 0 prefill plan: <P>t in <k> forward(s) of
        [<r0>,...,<rk-1>] rows, capacity <C>, <k*L> collectives of
        [<r0*H*4>B xL, ..., <rk-1*H*4>B xL]
    [c2-probe rank0 t=Nms] member 0 prefill complete: <P>t in <k> forward(s),
        <k*L> collectives
    [c2-probe rank0 t=Nms] member 0 prefill issued: <k*L> collectives of
        [<r0*H*4>B xL, ..., <rk-1*H*4>B xL]
  Read the three in that order, and read the byte RUNS rather than the counts
  alone:
    * one run of L collectives per chunk, chunk order preserved, and only the
      LAST run shorter than C. A single-chunk prompt is the degenerate case:
      one run of L collectives of P*H*4 bytes.
    * a decode forward is L collectives of H*4 bytes, so the member line's
      collectives= is (k + A) * L for A decode forwards.
    * a shortened final chunk is NORMAL and is not a desync; a different
      NUMBER of runs, or a different run length, is.
    * a 30 s gap between "prefill plan" and "prefill complete" is a stalled
      collective, not a slow prompt: the chunk count does not change the
      per-collective timeout.

Exit reasons:
  0 pass   1 transport/model/execution failure   2 arguments
  3 reserved (the unimplemented rank-1 role was retired)
  4 envelope mismatch   5 digest mismatch   6 member order mismatch
  7 worker C2 error response (the --worker-max-pending 1 and --mtp-model
    verdicts)
  8/9 member 0/1 exceeded its token budget
  10/11 member 0/1 tokens disagree with the local greedy output
  12 a C2 error response arrived, but not the MTP sidecar refusal that
    --expect-worker-mtp asserts (the mode still failed closed, for another
    reason: another refusal, or a changed production message)
  13 the prefill collective byte sequence rank 0 issued for a member differs
    from the one the prefill chunking predicted, so the executed and reported
    forward counts disagree and the lockstep would not mirror the worker
  the --tp-scope-override verdict is 1 on rank 0 and 1 on rank 1
  mode 3 also exits 0 on rank 1, like mode 1, and 7 on rank 0; a handshake
  disagreement (rank 1 --mtp-model without rank 0 --expect-worker-mtp, or
  mismatched --draft-tokens) is 1, because it fails before any command
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

/// Records the byte size of every collective rank 0 issues, while a recording
/// window is open.
///
/// `Model::Load` wires the executor's all-reduce callback to this object's
/// `ExchangePartial` (`Executor::TwoRankAllReduce`) and the callback is the
/// ONLY route from a forward to the wire, so the sizes collected here ARE the
/// byte sequence the peer sees. That is the point: the prefill chunking is a
/// prediction, and a prediction the probe also reports as its own result is a
/// prediction it can be wrong about silently. Recording makes the prediction
/// checkable, and a divergence is reported at the first offending index
/// instead of surfacing as an unattributable collective timeout.
///
/// Every method delegates, so the wire behaviour is exactly the inner
/// communicator's; the decorator adds no collective and drops none. No lock:
/// the executor invokes the callback on the caller's stream, and the probe
/// drives one model from one thread.
class CollectiveTrace : public q::rocm::Communicator {
public:
  explicit CollectiveTrace(std::shared_ptr<q::rocm::Communicator> inner)
      : inner_(std::move(inner)) {}

  [[nodiscard]] std::uint32_t rank() const noexcept override {
    return inner_->rank();
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept override {
    return inner_->world_size();
  }
  [[nodiscard]] int device_index() const noexcept override {
    return inner_->device_index();
  }
  [[nodiscard]] bool BeginOperation(std::uint64_t scope_id,
                                    std::string* error) override {
    return inner_->BeginOperation(scope_id, error);
  }
  [[nodiscard]] bool EndOperation(std::uint64_t scope_id,
                                  std::string* error) override {
    return inner_->EndOperation(scope_id, error);
  }
  [[nodiscard]] const float* ExchangePartial(const float* data,
                                             std::size_t bytes,
                                             hipStream_t stream,
                                             std::string* error) override {
    if (recording_) {
      sizes_.push_back(bytes);
    }
    if (!timing_) {
      return inner_->ExchangePartial(data, bytes, stream, error);
    }
    // Drain the stream first, untimed, so the timed call measures the
    // exchange alone and not the layer that produced its input. The inner
    // communicator synchronizes the same stream first anyway.
    if (bytes != 0 && hipStreamSynchronize(stream) != hipSuccess) {
      *error = "timed collective stream synchronize failed";
      return nullptr;
    }
    const auto start = std::chrono::steady_clock::now();
    const float* peer = inner_->ExchangePartial(data, bytes, stream, error);
    timed_micros_.push_back(std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - start)
                                .count());
    return peer;
  }

  /// Starts timing every collective, excluding the GPU work queued before it.
  void BeginTiming() {
    timed_micros_.clear();
    timing_ = true;
  }
  /// Stops timing and returns each collective's duration in microseconds.
  [[nodiscard]] std::vector<double> EndTiming() {
    timing_ = false;
    return std::move(timed_micros_);
  }

  /// Starts a recording window. The prefill is the only window the probe opens:
  /// it is the phase whose chunking is predicted, and its length is bounded by
  /// `ceil(context / PrefillCapacity) * num_layers` entries, so nothing here
  /// grows with the token budget.
  void BeginRecording() {
    sizes_.clear();
    recording_ = true;
  }
  /// Closes the window and returns the byte size of every collective issued
  /// inside it, in order.
  [[nodiscard]] std::vector<std::size_t> EndRecording() {
    recording_ = false;
    return sizes_;
  }

private:
  std::shared_ptr<q::rocm::Communicator> inner_;
  std::vector<std::size_t> sizes_;
  bool recording_{false};
  std::vector<double> timed_micros_;
  bool timing_{false};
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

/// Row counts of the forwards one `Session::Sync` issues for `prompt_tokens`.
///
/// `Session::Feed` advances by `exec.max_batch` in
/// `min(exec.max_batch, remaining)` row steps (engine.cpp:599-620) and
/// `Model::Load` sets `exec.max_batch = PrefillCapacity()` (engine.cpp:190), so
/// a fresh session prefills `P` tokens in exactly `ceil(P / PrefillCapacity)`
/// forwards. The worker's Qwen Flash-Next `Prefill` caps `consumed` at the same
/// `PrefillCapacity` and re-`Sync`s the cumulative prefix, whose common-prefix
/// skip feeds exactly those new tokens (inference_backend.cpp:2646-2672), so it
/// issues the same forwards with the same row counts in the same order. The
/// probe's runner pool holds one runner, so the lone resident prefill is
/// unbounded (`bounded_prefill` is false at
/// text_generation_scheduler.cpp:568-572) and no decode interleaves the chunks.
///
/// `capacity` is at least 1: it is `min(2048, max_context)` of a model that
/// loaded with the nonzero `--context` the probe validated. The loop advances
/// by `capacity`, so that is also the precondition for its termination.
std::vector<std::size_t> PrefillChunkRows(std::size_t prompt_tokens,
                                          std::uint32_t capacity) {
  std::vector<std::size_t> rows;
  for (std::size_t fed = 0; fed < prompt_tokens; fed += capacity) {
    rows.push_back(std::min<std::size_t>(capacity, prompt_tokens - fed));
  }
  return rows;
}

/// What the prefill chunking says one member will cost.
///
/// This is the PREDICTION, computed once from the token count and the model's
/// own `PrefillCapacity`, before the command is sent. `MemberTrace` carries the
/// observation; the two are compared in `main` and a divergence is exit 13.
struct MemberPlan {
  std::size_t prompt_tokens{0};
  std::uint32_t capacity{0};
  /// Rows per forward, in chunk order. Every entry but the last is `capacity`.
  std::vector<std::size_t> chunk_rows;
  /// `ceil(prompt_tokens / capacity)`, the number of prefill forwards.
  std::size_t prefill_forwards{0};
  /// The predicted byte size of every prefill collective, in order:
  /// `num_layers` copies of `rows * hidden_size * 4` per chunk.
  std::vector<std::size_t> expected_sizes;
};

MemberPlan PlanPrefill(std::size_t prompt_tokens, std::uint32_t capacity,
                       const q::Config& config) {
  MemberPlan plan;
  plan.prompt_tokens = prompt_tokens;
  plan.capacity = capacity;
  plan.chunk_rows = PrefillChunkRows(prompt_tokens, capacity);
  plan.prefill_forwards = plan.chunk_rows.size();
  plan.expected_sizes.reserve(plan.prefill_forwards * config.num_layers);
  const std::size_t row_bytes =
      static_cast<std::size_t>(config.hidden_size) * sizeof(float);
  for (const std::size_t rows : plan.chunk_rows) {
    for (std::size_t layer = 0; layer < config.num_layers; ++layer) {
      plan.expected_sizes.push_back(rows * row_bytes);
    }
  }
  return plan;
}

/// `2048, 952`, the per-forward row counts, comma separated.
std::string ChunkRowsText(const std::vector<std::size_t>& rows) {
  std::string text;
  for (const std::size_t value : rows) {
    if (!text.empty()) {
      text.push_back(',');
    }
    text += std::to_string(value);
  }
  return text.empty() ? "<none>" : text;
}

/// `65536B x48, 30464B x48`, the collective byte sequence run-length encoded.
///
/// Run length is what makes a chunked trace readable: one run of `num_layers`
/// collectives per forward, so the reader counts forwards and sees a shortened
/// final run directly, instead of comparing two long integer lists.
std::string ByteRunsText(const std::vector<std::size_t>& sizes) {
  std::string text;
  for (std::size_t index = 0; index < sizes.size();) {
    std::size_t run = index;
    while (run < sizes.size() && sizes[run] == sizes[index]) {
      ++run;
    }
    if (!text.empty()) {
      text += ", ";
    }
    text += std::to_string(sizes[index]) + "B x" + std::to_string(run - index);
    index = run;
  }
  return text.empty() ? "<none>" : text;
}

/// Index of the first position where the observed and predicted collective byte
/// sequences differ, or the shorter length when one is a prefix of the other.
std::size_t FirstSizeDifference(std::span<const std::size_t> expected,
                                std::span<const std::size_t> actual) {
  const std::size_t shared = std::min(expected.size(), actual.size());
  for (std::size_t index = 0; index < shared; ++index) {
    if (expected[index] != actual[index]) {
      return index;
    }
  }
  return shared;
}

/// The synthetic prompt `--prompt-tokens-N` asks for: exactly `count` ids,
/// deterministic, and different for each member.
///
/// A synthetic sequence is what makes a long-prompt run reproducible. A text
/// prompt's token count is a property of the vocabulary, so the same
/// `--prompt-0` can produce a different id count -- and therefore a different
/// chunk count, a different plan digest and a different byte sequence -- on any
/// vocabulary change, and the probe could no longer claim to have exercised a
/// multi-chunk prefill at all. The ids stay inside a low band of real
/// byte-level tokens so the embeddings are well conditioned, and an id the
/// model reports as a stop token is stepped over rather than dropped, because
/// the count is the contract.
std::vector<std::int32_t> SyntheticPrompt(const q::Model& model,
                                          std::size_t count,
                                          std::size_t member) {
  const std::size_t vocab = model.VocabSize();
  if (vocab <= static_cast<std::size_t>(kSyntheticFirstToken)) {
    // No usable id. An empty prompt is refused by the caller, which is the same
    // exit the tokenizer path reports for an empty prompt, so there is no
    // unrepresentable case left.
    return {};
  }
  // Clamped so the walk stays inside the embedding: the largest id it can
  // produce is `kSyntheticFirstToken + span - 1`, and `span <= vocab - 1`.
  const std::size_t span = std::min(kSyntheticSpan, vocab - 1);
  const std::size_t offset = member * kSyntheticMemberOffset;
  std::vector<std::int32_t> tokens;
  tokens.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    std::size_t id =
        kSyntheticFirstToken + ((index + offset) * kSyntheticStride) % span;
    // Step over a stop id rather than dropping the position: the count is the
    // contract. `id + 1 < vocab` keeps the walk from ever naming an id at or
    // past the end of the embedding.
    while (id + 1 < vocab && model.IsStopToken(static_cast<std::int32_t>(id))) {
      ++id;
    }
    tokens.push_back(static_cast<std::int32_t>(id));
  }
  return tokens;
}

/// The collective trace rank 0 issued for one member, kept for the report.
///
/// `prefill_sizes` is an OBSERVATION: the byte size of every collective
/// `CollectiveTrace` saw rank 0 issue during this member's prefill, in order.
/// `prefill_forwards` is derived from that observation and from nothing else,
/// so it can disagree with `MemberPlan::prefill_forwards` -- which is the
/// point.
struct MemberTrace {
  std::vector<std::int32_t> tokens;
  std::size_t prefill_forwards{0};
  std::size_t decode_forwards{0};
  std::vector<std::size_t> prefill_sizes;
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
/// The body mirrors one worker member: one `Sync` for the whole prompt, which
/// the model chunks into `ceil(P / PrefillCapacity)` forwards, then the
/// `SelectNext`/`Advance` alternation with the length-limit short circuit. The
/// prefill boundary is logged separately from the member completion so a stall
/// can be attributed to the prefill collective rather than to the decode
/// alternation or to model-load skew, and the prefill's byte sequence is
/// observed rather than assumed so the chunk arithmetic stays checkable.
bool RunMember(const char* role, std::size_t index,
               const std::shared_ptr<q::Model>& model, const MemberPlan& plan,
               std::span<const std::int32_t> prompt, std::uint32_t budget,
               std::uint32_t context, CollectiveTrace& collectives,
               MemberTrace* trace, std::string* error) {
  auto session = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      context, error);
  if (!session) {
    return false;
  }
  // Reported BEFORE the prefill: the whole point of the plan line is that a
  // reader holding a stalled run can see what was promised for the collectives
  // that never completed.
  Say(role, "member " + std::to_string(index) +
                " prefill plan: " + std::to_string(plan.prompt_tokens) +
                "t in " + std::to_string(plan.prefill_forwards) +
                " forward(s) of [" + ChunkRowsText(plan.chunk_rows) +
                "] rows, capacity " + std::to_string(plan.capacity) + ", " +
                std::to_string(plan.expected_sizes.size()) +
                " collectives of [" + ByteRunsText(plan.expected_sizes) + "]");
  collectives.BeginRecording();
  if (!session->Sync(prompt, error)) {
    // The window is still closed: a failed prefill is reported by the caller as
    // a transport failure, and leaving it open would fold a later phase's
    // collectives into this member's trace.
    (void)collectives.EndRecording();
    return false;
  }
  // The forward count is read off the observed sequence: a forward is exactly
  // `num_layers` collectives of one row count (executor.cpp:2013-2050), so the
  // count is the number of collectives divided by the layer count, and the
  // sizes are checked against the prediction afterwards rather than here.
  const std::size_t num_layers = model->config().num_layers;
  trace->prefill_sizes = collectives.EndRecording();
  trace->prefill_forwards =
      num_layers == 0 ? 0 : trace->prefill_sizes.size() / num_layers;
  Say(role, "member " + std::to_string(index) + " prefill complete: " +
                std::to_string(prompt.size()) + "t in " +
                std::to_string(trace->prefill_forwards) + " forward(s), " +
                std::to_string(trace->prefill_sizes.size()) + " collectives");
  Say(role, "member " + std::to_string(index) + " prefill issued: " +
                std::to_string(trace->prefill_sizes.size()) +
                " collectives of [" + ByteRunsText(trace->prefill_sizes) + "]");

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
    const auto sampled = static_cast<std::int32_t>(sampler.Sample(logits));
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

/// A 64-bit FNV-1a over raw token bytes, so a cross-rank comparison has one
/// token to diff instead of two id lists. Not a cryptographic digest: it only
/// has to detect an unintended difference.
/// `--moe-input-hashes`: hashes every MoE input while armed, so the ranks'
/// replicated hidden states can be compared layer by layer. Each hash drains
/// the stream and copies the input to the host, so it is armed only around
/// prefill.
struct MoeInputHashes {
  bool armed{false};
  std::vector<std::uint64_t> hashes;
  std::vector<std::size_t> sizes;
  std::vector<std::uint8_t> host;

  void Observe(const float* data, std::size_t bytes, hipStream_t stream) {
    if (!armed) {
      return;
    }
    host.resize(bytes);
    std::uint64_t hash = 0;
    if (hipStreamSynchronize(stream) == hipSuccess &&
        hipMemcpy(host.data(), data, bytes, hipMemcpyDeviceToHost) ==
            hipSuccess) {
      hash = 1469598103934665603ULL;
      for (const std::uint8_t byte : host) {
        hash ^= byte;
        hash *= 1099511628211ULL;
      }
    }
    hashes.push_back(hash);
    sizes.push_back(bytes);
  }
};

/// FNV-1a over the exact bytes of a logit row, so two ranks that compute the
/// same row bit for bit print the same value.
std::uint64_t LogitChecksum(std::span<const float> logits) {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(logits.data());
  for (std::size_t index = 0; index < logits.size_bytes(); ++index) {
    hash ^= bytes[index];
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::uint64_t TokenChecksum(std::span<const std::int32_t> tokens) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const std::int32_t token : tokens) {
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= static_cast<std::uint8_t>(
          (static_cast<std::uint32_t>(token) >> (8 * byte)) & 0xFFU);
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

std::string TokenListText(std::span<const std::int32_t> tokens) {
  std::string text;
  for (const std::int32_t token : tokens) {
    if (!text.empty()) {
      text += ',';
    }
    text += std::to_string(token);
  }
  return text;
}

/// Width-2 batched advance under TP=2, run symmetrically on BOTH ranks with no
/// cohort, no control wire and no worker. This is the step-2 spike: the serial
/// C2 contract can never exercise a two-row `EvaluateBatch`, so the question
/// this answers is whether a batched forward issues collectives both ranks
/// agree on -- same scope, same strictly monotonic ordinal, same byte count.
///
/// Both ranks run this identical program, so the log is the whole comparison:
/// each rank prints its per-member token list and checksum plus the exact
/// member set included in every batched step, and the harness diffs the two
/// logs. Schedule agreement is checked as well as token agreement, because a
/// member that stops early drops out of later batches -- and if the ranks ever
/// disagreed about that, the byte counts would diverge and the run would have
/// to fail closed rather than quietly continue.
///
/// `serial` is the baseline for the same program: each step advances the
/// included members one at a time instead of in one two-row forward, so a
/// batched and a serial run differ only in batching. Both report the decode
/// time spent in the advances alone, excluding sampling and logging.
int RunBatchedW2(const char* role, const std::shared_ptr<q::Model>& model,
                 const std::vector<std::int32_t> (&prompt)[kMemberCount],
                 std::uint32_t budget, std::uint32_t context,
                 const std::shared_ptr<CollectiveTrace>& collectives,
                 bool serial, MoeInputHashes* moe_inputs, std::string* error) {
  // Both ranks bind the same scope id. Nothing on the wire negotiates it in
  // this mode, so it is a constant agreed by construction, and the log prints
  // it so a reader can confirm both sides bound the same value.
  constexpr std::uint64_t kBatchedScope = 1;

  std::array<std::unique_ptr<q::Session>, kMemberCount> session;
  for (std::size_t index = 0; index < kMemberCount; ++index) {
    session[index] = model->CreateSession(
        gufo::core::SessionMode::kAutoregressive, context, error);
    if (!session[index]) {
      return kTransportFailure;
    }
  }

  OperationScope scope(collectives, kBatchedScope);
  if (!scope.Begin(error)) {
    Warn(std::string(role) + " could not bind batched scope " +
         std::to_string(kBatchedScope) + ": " + *error);
    return kTransportFailure;
  }
  Say(role,
      "batched scope bound: " + std::to_string(kBatchedScope) +
          ", prefill_capacity=" + std::to_string(model->PrefillCapacity()) +
          " budget=" + std::to_string(budget));

  // Prefill stays per-session and per-member: `Prefill` is per-state and has no
  // batch form in the runner or the engine, so a width-2 program chunks member
  // 0 and then member 1. Only the DECODE advance is batched here.
  MemberTrace trace[kMemberCount];
  // A zero-byte collective is a barrier: the ranks load the model at different
  // speeds, and without it the first prefill collective would bill the slower
  // load to the faster rank's prefill time.
  if (collectives->ExchangePartial(nullptr, 0, nullptr, error) == nullptr) {
    Warn(std::string(role) + " pre-prefill barrier failed: " + *error);
    return kTransportFailure;
  }
  const auto prefill_start = std::chrono::steady_clock::now();
  if (moe_inputs != nullptr) {
    moe_inputs->armed = true;
  }
  collectives->BeginRecording();
  for (std::size_t index = 0; index < kMemberCount; ++index) {
    if (!session[index]->Sync(prompt[index], error)) {
      (void)collectives->EndRecording();
      Warn(std::string(role) + " member " + std::to_string(index) +
           " batched prefill failed: " + *error);
      return kTransportFailure;
    }
  }
  if (moe_inputs != nullptr) {
    moe_inputs->armed = false;
    for (std::size_t index = 0; index < moe_inputs->hashes.size(); ++index) {
      Say(role, "moe-input " + std::to_string(index) + " bytes=" +
                    std::to_string(moe_inputs->sizes[index]) + " hash=" +
                    std::to_string(moe_inputs->hashes[index]));
    }
  }
  const auto prefill_elapsed = std::chrono::steady_clock::now() - prefill_start;
  const double prefill_ms =
      std::chrono::duration<double, std::milli>(prefill_elapsed).count();
  trace[0].prefill_sizes = collectives->EndRecording();
  const std::size_t num_layers = model->config().num_layers;
  trace[0].prefill_forwards =
      num_layers == 0 ? 0 : trace[0].prefill_sizes.size() / num_layers;
  Say(role, "batched prefill complete: " +
                std::to_string(trace[0].prefill_sizes.size()) +
                " collectives in " + std::to_string(trace[0].prefill_forwards) +
                " forward(s) of [" + ByteRunsText(trace[0].prefill_sizes) +
                "] rows, prompt sizes " + std::to_string(prompt[0].size()) +
                "/" + std::to_string(prompt[1].size()) + " tokens");
  const std::size_t prefill_tokens = prompt[0].size() + prompt[1].size();
  char prefill_timing[128];
  std::snprintf(prefill_timing, sizeof(prefill_timing),
                "prefill_ms=%.1f prefill_tokens=%zu prefill_tokens_per_s=%.1f",
                prefill_ms, prefill_tokens,
                prefill_ms <= 0.0 ? 0.0 : prefill_tokens * 1000.0 / prefill_ms);
  Say(role, std::string("prefill timing: ") + prefill_timing);

  const gufo::sampling::SamplingConfig sampling{
      .temperature = kGreedyTemperature,
  };
  gufo::sampling::SamplerState sampler0(sampling);
  gufo::sampling::SamplerState sampler1(sampling);
  gufo::sampling::SamplerState* samplers[kMemberCount] = {&sampler0, &sampler1};

  // Mirrors `RunMember` step for step, with the per-member advance replaced by
  // one batched advance. `RunMember` also stops before evaluating the token
  // that reaches the budget, because the distributed runner sets
  // `final_token_advance_required` false; the same short circuit is kept so the
  // two programs issue the same number of forwards.
  std::chrono::steady_clock::duration decode_time{};
  std::size_t timed_steps = 0;
  collectives->BeginTiming();
  std::size_t forwards = 0;
  std::size_t advanced_tokens = 0;
  for (std::size_t step = 0; step < budget; ++step) {
    std::array<q::Session::AdvanceRequest, kMemberCount> requests;
    std::array<std::size_t, kMemberCount> members{};
    std::size_t rows = 0;
    std::string included;
    std::string logit_hashes;
    for (std::size_t index = 0; index < kMemberCount; ++index) {
      if (!trace[index].stopped_on_token &&
          trace[index].tokens.size() >= budget) {
        trace[index].stopped_on_budget = true;
      }
      if (trace[index].stopped_on_token || trace[index].stopped_on_budget) {
        continue;
      }
      const auto logits = session[index]->Logits();
      if (logits.empty()) {
        *error = "session " + std::to_string(index) +
                 " has no logits to sample from";
        return kTransportFailure;
      }
      // The logits a rank samples from: ranks that agree bit for bit here
      // hold the same hidden state, so the first differing step localizes a
      // numerical divergence even before any sampled token differs.
      logit_hashes += (logit_hashes.empty() ? "" : ",") +
                      std::to_string(LogitChecksum(logits));
      const auto sampled =
          static_cast<std::int32_t>(samplers[index]->Sample(logits));
      if (model->IsStopToken(sampled)) {
        trace[index].stopped_on_token = true;
        continue;
      }
      trace[index].tokens.push_back(sampled);
      if (trace[index].tokens.size() >= budget) {
        trace[index].stopped_on_budget = true;
        continue;
      }
      requests[rows] = q::Session::AdvanceRequest{
          .session = session[index].get(),
          .token = sampled,
      };
      members[rows] = index;
      included += static_cast<char>('a' + index);
      ++rows;
    }
    // The per-step member set is the schedule. It must be identical on both
    // ranks, so it is logged rather than inferred.
    Say(role, "batched step " + std::to_string(step) + " rows=" +
                  std::to_string(rows) + " members=" +
                  (included.empty() ? std::string("none") : included) +
                  " logits=" + logit_hashes);
    if (rows == 0) {
      break;
    }
    const auto step_start = std::chrono::steady_clock::now();
    if (serial) {
      for (std::size_t row = 0; row < rows; ++row) {
        if (!requests[row].session->Evaluate(requests[row].token, error)) {
          Warn(std::string(role) + " serial advance failed at step " +
               std::to_string(step) + ": " + *error);
          return kTransportFailure;
        }
      }
    } else if (!q::Session::EvaluateBatch(std::span(requests).first(rows),
                                          error)) {
      Warn(std::string(role) + " batched advance failed at step " +
           std::to_string(step) + ": " + *error);
      return kTransportFailure;
    }
    decode_time += std::chrono::steady_clock::now() - step_start;
    ++timed_steps;
    forwards += serial ? rows : 1;
    advanced_tokens += rows;
    // Only the members in this step's advance took part in a forward.
    for (std::size_t row = 0; row < rows; ++row) {
      ++trace[members[row]].decode_forwards;
    }
  }
  std::vector<double> collective_micros = collectives->EndTiming();
  const double decode_ms =
      std::chrono::duration<double, std::milli>(decode_time).count();
  char timing[160];
  std::snprintf(timing, sizeof(timing),
                "decode_ms=%.1f ms_per_step=%.2f ms_per_forward=%.2f "
                "tokens_per_s=%.1f",
                decode_ms, timed_steps == 0 ? 0.0 : decode_ms / timed_steps,
                forwards == 0 ? 0.0 : decode_ms / forwards,
                decode_ms <= 0.0 ? 0.0 : advanced_tokens * 1000.0 / decode_ms);
  Say(role, std::string(serial ? "serial" : "batched") +
                " decode: steps=" + std::to_string(timed_steps) +
                " forwards=" + std::to_string(forwards) + " advanced_tokens=" +
                std::to_string(advanced_tokens) + " " + timing);
  if (!collective_micros.empty()) {
    std::sort(collective_micros.begin(), collective_micros.end());
    double collective_total = 0.0;
    for (const double value : collective_micros) {
      collective_total += value;
    }
    char collective_line[192];
    std::snprintf(collective_line, sizeof(collective_line),
                  "decode collectives: n=%zu total_ms=%.1f per_forward_ms=%.2f "
                  "p50_us=%.1f p90_us=%.1f p99_us=%.1f",
                  collective_micros.size(), collective_total / 1000.0,
                  forwards == 0 ? 0.0 : collective_total / 1000.0 / forwards,
                  collective_micros[collective_micros.size() / 2],
                  collective_micros[(collective_micros.size() * 9) / 10],
                  collective_micros[(collective_micros.size() * 99) / 100]);
    Say(role, collective_line);
  }

  if (!scope.End(error)) {
    Warn(std::string(role) + " could not release batched scope: " + *error);
    return kTransportFailure;
  }
  for (std::size_t index = 0; index < kMemberCount; ++index) {
    Say(role,
        "batched member " + std::to_string(index) +
            " tokens=" + TokenListText(trace[index].tokens) +
            " count=" + std::to_string(trace[index].tokens.size()) +
            " checksum=" + std::to_string(TokenChecksum(trace[index].tokens)) +
            " decode_forwards=" + std::to_string(trace[index].decode_forwards) +
            (trace[index].stopped_on_token ? " stop_token" : "") +
            (trace[index].stopped_on_budget ? " budget" : ""));
  }
  return kOk;
}

/// `--allreduce-bench N`: times N back-to-back all-reduces per payload with no
/// model loaded, on both ranks in lockstep. The payloads are one, two and eight
/// Flash-Next decode rows and one 512-token prefill chunk, so the result is the
/// bare cost of a collective as the model issues it: the exchange (stream sync,
/// device-to-host staging, the TCP header, the RDMA read and the ack) plus the
/// GPU add of the peer's partial, timed until the add completes. Model compute
/// is absent by construction.
int RunAllReduceBench(const char* role, q::rocm::Communicator& communicator,
                      std::uint32_t device, std::uint32_t iterations) {
  // One Flash-Next hidden row: 2560 floats, the decode all-reduce unit.
  constexpr std::size_t kRowBytes = 2560 * sizeof(float);
  constexpr std::array<std::size_t, 4> kRows{1, 2, 8, 512};
  constexpr std::uint64_t kBenchScope = 1;
  if (hipSetDevice(static_cast<int>(device)) != hipSuccess) {
    Warn(std::string(role) + " allreduce bench could not select the device");
    return kTransportFailure;
  }
  float* data = nullptr;
  hipStream_t stream = nullptr;
  const std::size_t max_bytes = kRows.back() * kRowBytes;
  if (hipMalloc(reinterpret_cast<void**>(&data), max_bytes) != hipSuccess ||
      hipMemset(data, 0, max_bytes) != hipSuccess ||
      hipStreamCreate(&stream) != hipSuccess) {
    Warn(std::string(role) + " allreduce bench could not allocate buffers");
    if (data != nullptr) {
      (void)hipFree(data);
    }
    return kTransportFailure;
  }
  const auto release = [&] {
    (void)hipStreamDestroy(stream);
    (void)hipFree(data);
  };
  std::string error;
  if (!communicator.BeginOperation(kBenchScope, &error)) {
    Warn(std::string(role) + " allreduce bench scope bind failed: " + error);
    release();
    return kTransportFailure;
  }
  for (const std::size_t rows : kRows) {
    const std::size_t bytes = rows * kRowBytes;
    std::vector<double> micros;
    micros.reserve(iterations);
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
      const auto start = std::chrono::steady_clock::now();
      const float* peer =
          communicator.ExchangePartial(data, bytes, stream, &error);
      if (peer == nullptr) {
        Warn(std::string(role) + " allreduce bench failed at " +
             std::to_string(bytes) + " B: " + error);
        release();
        return kTransportFailure;
      }
      q::rocm::AddRowsBroadcast(peer, data, static_cast<std::uint32_t>(rows),
                                kRowBytes / sizeof(float), 1, stream);
      if (hipStreamSynchronize(stream) != hipSuccess) {
        Warn(std::string(role) + " allreduce bench add failed");
        release();
        return kTransportFailure;
      }
      micros.push_back(std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - start)
                           .count());
    }
    std::sort(micros.begin(), micros.end());
    double total = 0.0;
    for (const double value : micros) {
      total += value;
    }
    char line[192];
    std::snprintf(line, sizeof(line),
                  "allreduce rows=%zu bytes=%zu n=%u min_us=%.1f p50_us=%.1f "
                  "mean_us=%.1f p99_us=%.1f",
                  rows, bytes, iterations, micros.front(),
                  micros[micros.size() / 2], total / micros.size(),
                  micros[(micros.size() * 99) / 100]);
    Say(role, line);
  }
  if (!communicator.EndOperation(kBenchScope, &error)) {
    Warn(std::string(role) + " allreduce bench scope release failed: " + error);
    release();
    return kTransportFailure;
  }
  release();
  return kOk;
}

/// The rank-one MTP draft sidecar for the third fault mode.
///
/// An empty path is the production no-MTP policy, and `draft_tokens` is only
/// read with a path, so a sidecar-less run always builds the same
/// `TextSpeculativeConfig` the PASS path has always used.
struct MtpSidecar {
  std::string model_path;
  std::uint32_t draft_tokens{q::kMaxMtpDraftTokens};

  [[nodiscard]] bool loaded() const { return !model_path.empty(); }
};

/// Hosts the real rank-1 C2 worker and returns a probe exit reason.
///
/// `gufo serve` cannot be reused for this: it forces `--max-pending 1` for
/// TP=2, so `SubmitCohort` refuses every cohort at admission and answers a
/// silent C2 error without running a single collective. The probe therefore
/// builds `InferenceBackend` itself and supplies the policy the cohort needs.
/// Nothing here relaxes a production guard.
///
/// `mtp` is the third fault mode: a real MTP draft sidecar, so the worker's
/// handshake presents a draft sidecar and `RunTpCohortCommand` refuses the
/// cohort before it binds a scope. Empty is the production no-MTP policy.
int RunRank1Worker(const std::string& model_path, std::uint32_t context,
                   std::uint32_t device, std::uint32_t gid,
                   const std::string& bootstrap_host,
                   std::uint16_t bootstrap_port, std::uint16_t control_port,
                   const std::string& control_token,
                   std::uint32_t worker_max_pending, const MtpSidecar& mtp) {
  const bool mtp_loaded = mtp.loaded();
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
      // `--worker-max-pending` moves this ceiling and nothing else: 2 admits
      // the cohort, 1 provokes the `SubmitCohort` refusal documented in the
      // file header. The runner pool capacity comes from `kWorkerSessionCount`,
      // not from this field, so the collective trace is unchanged.
      .max_pending_requests = worker_max_pending,
      // Rank 0 sends one member per client id, so each member is admitted on
      // its own and the per-client ceiling never blocks the cohort.
      .max_pending_requests_per_client = kWorkerMaxPendingRequestsPerClient,
  };
  // Without a sidecar: backend kDisabled with an empty draft path, so the model
  // loads no MTP sidecar, the handshake reports 0 draft tokens and
  // `use_mtp = false`. The worker refuses a cohort while MTP is loaded
  // (tp_cohort_worker.cpp), and the C2 response contract cannot carry draft
  // telemetry. `max_draft_tokens` stays at its default 7 rather than 0:
  // `Model::Load` rejects a zero draft limit, and the handshake's draft field
  // comes from `has_mtp`, not from it.
  //
  // With a sidecar, this is exactly what `gufo serve --speculative mtp
  // --mtp-model <path> --draft-tokens N` builds (serve.cpp:1132-1174): backend
  // kMtp, the sidecar path, a positive draft limit and min_draft_tokens 1,
  // which `InferenceBackend::load` requires of Flash-Next MTP
  // (inference_backend.cpp:3418-3425 and 3790-3797). `max_draft_tokens` is
  // also the value the handshake presents (inference_backend.cpp:3845), so it
  // must equal rank 0's --draft-tokens.
  const server::TextSpeculativeConfig speculative_config{
      .backend = mtp_loaded ? server::TextSpeculativeBackend::kMtp
                            : server::TextSpeculativeBackend::kDisabled,
      .draft_model_path = mtp_loaded ? mtp.model_path : std::string{},
      .max_draft_tokens = mtp.draft_tokens,
      .min_draft_tokens = 1,
  };
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
  // `gufo serve` rejects `max_pending == 1` for TP=2 as part of its "TP2
  // currently requires C1" guard (serve.cpp:1210-1219). That guard is
  // deliberately NOT reproduced here: the probe bypasses the serving front door
  // and hosts `InferenceBackend` in-process, so `--worker-max-pending 1`
  // reaches the scheduler and provokes the real admission refusal. Forcing that
  // refusal is the entire point of the mode, and nothing here relaxes a
  // production guard or changes any file outside this test binary.
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
  // The ready line states the worker's speculative posture because it is part
  // of the handshake, so a reader can tell mode 3 from the PASS path without
  // cross-referencing the fault banner.
  std::string mtp_state = "no MTP";
  if (mtp_loaded) {
    mtp_state = "MTP draft sidecar " + mtp.model_path + " with " +
                std::to_string(mtp.draft_tokens) + " draft tokens";
  }
  Say("rank1",
      "ready: model loaded, TP control handshaken, runner pool capacity " +
          std::to_string(kWorkerSessionCount) + ", max_pending_requests " +
          std::to_string(worker_max_pending) + ", " + mtp_state +
          "; waiting for the C2 command");
  if (mtp_loaded) {
    // The refusal is decided before `BuildCohortMembers` and before admission
    // (tp_cohort_worker.cpp:164-166), so a loaded sidecar decides the outcome
    // even when `--worker-max-pending 1` is also set, and the two banners can
    // never both be true at once.
    Say("rank1",
        "FAULT INJECTED: --mtp-model " + mtp.model_path + " with " +
            std::to_string(mtp.draft_tokens) +
            " draft tokens loads a real MTP draft sidecar, so this worker's "
            "handshake presents use_mtp=true and max_draft_tokens=" +
            std::to_string(mtp.draft_tokens) +
            "; the next kCohort2Ar command must be refused BEFORE "
            "BeginOperation with error \"C2 cohort AR is not supported while "
            "the MTP sidecar is loaded\", so rank 1 must bind no lease, issue "
            "no collective and answer with a wire-valid C2 error response that "
            "still carries both member envelopes");
  } else if (worker_max_pending < 2) {
    // `queued_count + 2 <= max_pending_requests` cannot hold, so the next
    // cohort is refused at admission. Announce the injected fault so the log
    // is self-describing and no reader can mistake the refusal for a contract
    // regression.
    Say("rank1",
        "FAULT INJECTED: --worker-max-pending " +
            std::to_string(worker_max_pending) +
            " cannot admit a two-member cohort; the next kCohort2Ar command "
            "must be refused at admission with a C2 error response and no "
            "collective");
  }

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
  /// Whether `--prompt-N` was given for a member, so a default text prompt is
  /// not mistaken for a requested one when `--prompt-tokens-N` is also present.
  bool prompt_given[kMemberCount] = {false, false};
  /// `--prompt-tokens-N`: synthesize this member's prompt as exactly N token
  /// ids instead of tokenizing text. This is the only way to ask for a prompt
  /// whose length is known before the model is loaded, which is what makes a
  /// multi-chunk prefill exact and its context check free.
  std::optional<std::uint32_t> prompt_tokens[kMemberCount];
  std::uint64_t cohort_id = 1;
  std::uint64_t sequence = 1;
  std::uint64_t member_id[kMemberCount] = {1, 2};
  std::uint32_t budget = 8;
  std::uint32_t context = 4096;
  std::uint32_t device = 0;
  std::uint32_t gid = 0;
  /// Rank-one admission ceiling; see `--worker-max-pending`.
  std::uint32_t worker_max_pending = kWorkerMaxPendingRequests;
  /// Rank-zero fault injection: bind a collective scope that is not the command
  /// sequence. Unset in every non-fault run.
  std::optional<std::uint64_t> scope_override;
  /// Rank-zero fault injection: the worker will refuse the cohort at admission,
  /// so issue no lockstep collective and wait only for its response.
  bool expect_c2_error = false;
  /// `--batched-w2`: step-2 spike. Both ranks run the identical batched
  /// program over the engine with no cohort, no control wire and no worker, so
  /// the only variable is whether a two-row `EvaluateBatch` keeps both ranks on
  /// one collective schedule.
  bool batched_w2 = false;
  /// `--serial-w2`: the serial baseline for `--batched-w2`. Same sessions,
  /// prompts, scope and stopping rules, but each step advances the included
  /// members one at a time, so the two runs differ only in batching.
  bool serial_w2 = false;
  /// `--allreduce-bench N`: time N collectives per payload with no model.
  std::uint32_t allreduce_bench = 0;
  /// `--moe-input-hashes`: print a hash of every MoE input during prefill.
  bool moe_input_hashes = false;
  /// Rank-one fault injection: the MTP draft sidecar to load before the
  /// handshake. Empty is the production no-MTP policy.
  std::string mtp_model_path;
  /// MTP draft width, on both roles: rank1 loads it with `--mtp-model`, rank 0
  /// declares the worker's handshake value with `--expect-worker-mtp`. Unset
  /// means `kMaxMtpDraftTokens`, which is also what a sidecar-less worker
  /// passes to `load`.
  std::optional<std::uint32_t> draft_tokens;
  /// Rank-zero fault injection: rank 1's handshake presents a draft sidecar,
  /// so rank 0 mirrors `use_mtp` and the draft width without loading one.
  bool expect_worker_mtp = false;
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
      prompt_given[0] = true;
    } else if (arg == "--prompt-1") {
      prompt[1] = next();
      prompt_given[1] = true;
    } else if (arg == "--prompt-file-0") {
      prompt_file[0] = next();
    } else if (arg == "--prompt-file-1") {
      prompt_file[1] = next();
    } else if (arg == "--prompt-tokens-0" || arg == "--prompt-tokens-1") {
      // One parse branch for both members: the value is a token COUNT, so the
      // name carries the member and nothing else in the argument is read.
      const std::size_t index = arg == "--prompt-tokens-0" ? 0 : 1;
      std::uint32_t parsed = 0;
      if (!ParseUint(next(), &parsed)) {
        // A missing, negative, trailing-garbage or non-numeric value all land
        // here. None of them is a count, and none may fall back to the text
        // prompt: that would silently prefill a length nobody asked for.
        Warn(arg + " requires an unsigned token count");
        return kInvalidArguments;
      }
      if (parsed == 0) {
        // Refused rather than clamped: a zero-length prompt is refused by
        // `Session::Sync` and by `ValidateMemberRequest`, and a silent clamp
        // would report a prefill of a length nobody asked for.
        Warn(arg +
             " must be at least 1 token; a zero-token prompt is refused "
             "by both Sync and the C2 member validation");
        return kInvalidArguments;
      }
      prompt_tokens[index] = parsed;
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
    } else if (arg == "--worker-max-pending") {
      if (!ParseUint(next(), &worker_max_pending)) {
        Warn("--worker-max-pending requires an unsigned integer");
        return kInvalidArguments;
      }
    } else if (arg == "--tp-scope-override") {
      std::uint64_t parsed = 0;
      if (!ParseUint64(next(), &parsed)) {
        Warn("--tp-scope-override requires an unsigned integer");
        return kInvalidArguments;
      }
      scope_override = parsed;
    } else if (arg == "--expect-c2-error") {
      expect_c2_error = true;
    } else if (arg == "--batched-w2") {
      batched_w2 = true;
    } else if (arg == "--serial-w2") {
      serial_w2 = true;
    } else if (arg == "--moe-input-hashes") {
      moe_input_hashes = true;
    } else if (arg == "--allreduce-bench") {
      if (!ParseUint(next(), &allreduce_bench) || allreduce_bench == 0) {
        Warn("--allreduce-bench requires a positive iteration count");
        return kInvalidArguments;
      }
    } else if (arg == "--mtp-model") {
      // An absent or empty value is a missing argument, NOT the no-sidecar
      // default, so it is refused here instead of silently dropping the fault.
      mtp_model_path = next();
      if (mtp_model_path.empty()) {
        Warn("--mtp-model requires a path to an MTP draft sidecar");
        return kInvalidArguments;
      }
    } else if (arg == "--draft-tokens") {
      std::uint32_t parsed = 0;
      if (!ParseUint(next(), &parsed)) {
        Warn("--draft-tokens requires an unsigned integer");
        return kInvalidArguments;
      }
      draft_tokens = parsed;
    } else if (arg == "--expect-worker-mtp") {
      expect_worker_mtp = true;
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
  // In the cohort modes only rank 0 owns prompts, so only rank 0 reads prompt
  // files: rank 1 has no cohort to build and never issues a request of its
  // own. The two-program modes run the same prompts on both ranks.
  if (role == "rank0" || batched_w2 || serial_w2) {
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
  // The budget against the context, on BOTH roles and here rather than after
  // the model load: a budget that cannot fit the context is an argument
  // mistake, and discovering it after a 30 s RDMA rendezvous and a 30 s model
  // load wastes both. It is also the outer bound of the prompt-plus-budget
  // check below, so it is checked once, in the cheapest place.
  if (budget > context) {
    Warn("--max-tokens " + std::to_string(budget) +
         " exceeds the negotiated --context " + std::to_string(context));
    return kInvalidArguments;
  }
  if (cohort_id == 0 || sequence == 0) {
    Warn("--cohort-id and --sequence must both be nonzero");
    return kInvalidArguments;
  }
  // Rank-one policy, validated on BOTH roles so a typo cannot be discovered
  // only after a 30 s RDMA rendezvous. Zero would additionally be rejected by
  // the scheduler's own `max_pending_requests == 0` limit, and 1 is the
  // deliberate admission-refusal fault value, so only zero is refused here.
  if (worker_max_pending == 0) {
    Warn(
        "--worker-max-pending must be at least 1; 1 deliberately refuses a "
        "two-member cohort at admission and 2 is the admitting default");
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
  if (static_cast<int>(batched_w2) + static_cast<int>(serial_w2) +
          static_cast<int>(allreduce_bench != 0) >
      1) {
    Warn("--batched-w2, --serial-w2 and --allreduce-bench are exclusive");
    return kInvalidArguments;
  }
  if (device > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      gid > std::numeric_limits<std::uint8_t>::max()) {
    Warn("--tp-device or --tp-gid-index is out of range");
    return kInvalidArguments;
  }
  // Rank 0 is the only role that builds a cohort, so only rank 0 rejects an
  // empty prompt and only rank 0 can inject the scope override.
  if (role == "rank0") {
    for (std::size_t index = 0; index < kMemberCount; ++index) {
      const std::string option = "--prompt-tokens-" + std::to_string(index);
      // Two sources for one member's prompt is an argument mistake, not a
      // precedence question: whichever won silently, the run would exercise a
      // prompt the operator did not ask for and report a verdict about it.
      if (prompt_tokens[index].has_value() &&
          (prompt_given[index] || !prompt_file[index].empty())) {
        Warn(option + " cannot be combined with --prompt-" +
             std::to_string(index) + " or --prompt-file-" +
             std::to_string(index) +
             "; a member has exactly one prompt, and only the token count is "
             "exact before the model is loaded");
        return kInvalidArguments;
      }
      // A synthetic prompt's length is known here, so the context check runs
      // BEFORE any RDMA or control peer is touched: an over-long request costs
      // 0 ms instead of a 30 s rendezvous and a 30 s model load. A text or
      // file prompt cannot be checked here and is checked after the load, where
      // its token count first exists.
      if (prompt_tokens[index].has_value() &&
          static_cast<std::uint64_t>(*prompt_tokens[index]) + budget >
              context) {
        Warn("member " + std::to_string(index) + " " + option + " " +
             std::to_string(*prompt_tokens[index]) + " plus --max-tokens " +
             std::to_string(budget) + " (" +
             std::to_string(static_cast<std::uint64_t>(*prompt_tokens[index]) +
                            budget) +
             ") exceeds the negotiated --context " + std::to_string(context));
        return kInvalidArguments;
      }
      if (!prompt_tokens[index].has_value() && prompt[index].empty()) {
        Warn("member " + std::to_string(index) + " has an empty prompt");
        return kInvalidArguments;
      }
    }
    // The override must be a real disagreement, never a silent no-op: an
    // override equal to the command sequence would bind the same scope as the
    // worker and the run would report a PASS while injecting nothing, which is
    // exactly the false verdict this fault mode must not be able to produce.
    // Zero is refused for the same reason: it is the degenerate scope value
    // that stands for "no scope", and `BeginOperation` models "unbound" as a
    // null optional rather than as scope 0.
    if (scope_override.has_value() && *scope_override == 0) {
      Warn(
          "--tp-scope-override must be nonzero; the bound scope and the "
          "command sequence must be two distinct nonzero values");
      return kInvalidArguments;
    }
    if (scope_override.has_value() && *scope_override == sequence) {
      Warn("--tp-scope-override must differ from --sequence (" +
           std::to_string(sequence) +
           "); an equal override injects no fault and would report a PASS");
      return kInvalidArguments;
    }
    // The sidecar-refusal mode needs rank 0 to issue no collective at all,
    // because the whole point of the mode is that rank 1 refuses before
    // `BeginOperation`. Without `--expect-c2-error` rank 0 would run its
    // lockstep prefill, which nobody answers, and the run would end in a 30 s
    // collective timeout with both communicators poisoned instead of the C2
    // error response this mode exists to observe.
    if (expect_worker_mtp && !expect_c2_error) {
      Warn(
          "--expect-worker-mtp requires --expect-c2-error: the worker refuses "
          "the cohort before it binds a scope, so rank 0 must issue no "
          "collective and wait only for the C2 error response");
      return kInvalidArguments;
    }
  }

  // In the cohort modes the synthetic prompt count is a rank-zero concept: rank
  // 1 serves whatever rank 0 sends and never builds a cohort, so the flag would
  // be inert there. Refused rather than ignored, for the same reason
  // `--mtp-model` is refused on rank 0. The two-program modes are the
  // exception: each rank builds its own copy of the same prompts.
  if (role == "rank1" && !batched_w2 && !serial_w2 &&
      (prompt_tokens[0].has_value() || prompt_tokens[1].has_value())) {
    Warn(
        "--prompt-tokens-0/1 are rank-zero options: rank 1 hosts the worker "
        "and "
        "only serves the cohort rank 0 sends, so it has no prompt of its own "
        "to synthesize");
    return kInvalidArguments;
  }

  // Sidecar fault injection, validated on BOTH roles and before any RDMA or
  // control peer is touched, so a wrong sidecar path or an unpaired parity flag
  // costs 0 ms instead of a 30 s rendezvous and a model load. Rank 0 refuses
  // `--mtp-model` outright: it is the AR lockstep driver, the mode exists to
  // prove that no collective is issued, and a sidecar at rank 0 would also make
  // its own handshake disagree with the no-MTP model it loaded. The worker's
  // parity is declared with `--expect-worker-mtp` instead.
  if (!mtp_model_path.empty() && role == "rank0") {
    Warn(
        "--mtp-model is a rank-one option: rank 0 drives the AR lockstep and "
        "must load no draft sidecar, so declare the worker's parity with "
        "--expect-worker-mtp --expect-c2-error instead");
    return kInvalidArguments;
  }
  if (!mtp_model_path.empty()) {
    std::error_code filesystem_error;
    if (!std::filesystem::is_regular_file(std::filesystem::path(mtp_model_path),
                                          filesystem_error)) {
      Warn("--mtp-model is not a readable file: " + mtp_model_path);
      return kInvalidArguments;
    }
  }
  // `Model::Load` refuses a zero draft limit and caps Flash-Next MTP at
  // `kMaxMtpDraftTokens` (engine.cpp:111-119), and `Handshake` rejects
  // `use_mtp` with no draft tokens (tp_control.cpp:618-622), so the width is
  // checked here rather than after the rendezvous.
  if (draft_tokens.has_value() &&
      (*draft_tokens == 0 || *draft_tokens > q::kMaxMtpDraftTokens)) {
    Warn("--draft-tokens must be between 1 and " +
         std::to_string(q::kMaxMtpDraftTokens) +
         "; a zero draft limit is rejected by the model load and more than " +
         std::to_string(q::kMaxMtpDraftTokens) +
         " draft tokens by Flash-Next MTP");
    return kInvalidArguments;
  }
  // A width alone injects nothing: rank 1 only loads it with `--mtp-model` and
  // rank 0 only declares it with `--expect-worker-mtp`. An unpaired width is
  // refused rather than ignored, for the same reason `--tp-scope-override`
  // equal to `--sequence` is refused above.
  const bool sidecar_fault = (role == "rank1" && !mtp_model_path.empty()) ||
                             (role == "rank0" && expect_worker_mtp);
  if (draft_tokens.has_value() && !sidecar_fault) {
    Warn(
        "--draft-tokens needs --mtp-model (rank1) or --expect-worker-mtp "
        "(rank0); on this role it injects no fault and changes nothing");
    return kInvalidArguments;
  }
  // The width both roles present in the handshake: rank 1 loads it, rank 0
  // declares it. Unset keeps `kMaxMtpDraftTokens`, which is also what the
  // sidecar-less worker has always passed to `load`.
  const std::uint32_t mtp_draft_tokens =
      draft_tokens.value_or(q::kMaxMtpDraftTokens);

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
    // `--batched-w2` returns before the control connect, mirroring rank 0's
    // branch before its control listen. RDMA first and for the same reason as
    // `RunRank1Worker`: the bootstrap connect blocks for up to 30 s while rank
    // 0 listens, so the model must not be mapped before it completes.
    if (allreduce_bench != 0) {
      std::string bench_error;
      const q::rocm::IbrverbsConfig bench_rdma{
          .rank = kRank1,
          .world_size = kWorldSize,
          .bootstrap_host = bootstrap_host,
          .bootstrap_port = bootstrap_port,
          .device_index = device,
          .gid_index = gid,
      };
      auto communicator =
          q::rocm::CreateIbrverbsCommunicator(bench_rdma, &bench_error);
      if (!communicator) {
        Warn("allreduce bench RDMA communicator failed: " + bench_error);
        return kTransportFailure;
      }
      return RunAllReduceBench("rank1", *communicator, device,
                               allreduce_bench);
    }
    if (batched_w2 || serial_w2) {
      std::string batched_error;
      Say("rank1", std::string(serial_w2 ? "serial-w2" : "batched-w2") +
                       " mode: no control wire, no cohort, no worker");
      const q::rocm::IbrverbsConfig batched_rdma{
          .rank = kRank1,
          .world_size = kWorldSize,
          .bootstrap_host = bootstrap_host,
          .bootstrap_port = bootstrap_port,
          .device_index = device,
          .gid_index = gid,
      };
      auto communicator =
          q::rocm::CreateIbrverbsCommunicator(batched_rdma, &batched_error);
      if (!communicator) {
        Warn("batched-w2 RDMA communicator failed: " + batched_error);
        return kTransportFailure;
      }
      Say("rank1", "RDMA peer established with " + bootstrap_host +
                       " on bootstrap port " + std::to_string(bootstrap_port));
      auto collectives =
          std::make_shared<CollectiveTrace>(std::move(communicator));
      q::ModelOptions batched_options{
          .max_context = context,
          .mtp_model_path = "",
          .max_draft_tokens = q::kMaxMtpDraftTokens,
          .vision_model_path = "",
          .decode_concurrency = 1,
          .tp_rank = kRank1,
          .tp_world_size = kWorldSize,
          .hip_device = static_cast<int>(device),
          .communicator = collectives,
      };
      auto moe_inputs = std::make_shared<MoeInputHashes>();
      if (moe_input_hashes) {
        batched_options.moe_observer =
            [moe_inputs](const float* data, std::size_t bytes,
                         hipStream_t stream) {
              moe_inputs->Observe(data, bytes, stream);
            };
      }
      auto batched_model =
          q::Model::Load(model_path, batched_options, &batched_error);
      if (!batched_model) {
        Warn("batched-w2 model load failed: " + batched_error);
        return kTransportFailure;
      }
      if (batched_model->HasMtp()) {
        Warn(
            "batched-w2 model reports an MTP sidecar, which this mode refuses");
        return kInvalidArguments;
      }
      std::vector<std::int32_t> batched_prompt[kMemberCount];
      for (std::size_t index = 0; index < kMemberCount; ++index) {
        batched_prompt[index] =
            prompt_tokens[index].has_value()
                ? SyntheticPrompt(*batched_model, *prompt_tokens[index], index)
                : batched_model->Tokenize(prompt[index]);
        if (batched_prompt[index].empty()) {
          Warn("batched-w2 member " + std::to_string(index) +
               " prompt tokenized empty");
          return kInvalidArguments;
        }
        if (batched_prompt[index].size() + budget >
            static_cast<std::size_t>(context)) {
          Warn("batched-w2 member " + std::to_string(index) + " prompt has " +
               std::to_string(batched_prompt[index].size()) +
               " tokens, plus --max-tokens " + std::to_string(budget) +
               ", which exceeds the negotiated context " +
               std::to_string(context));
          return kInvalidArguments;
        }
      }
      return RunBatchedW2("rank1", batched_model, batched_prompt, budget,
                           context, collectives, serial_w2,
                           moe_input_hashes ? moe_inputs.get() : nullptr,
                           &batched_error);
    }
    const MtpSidecar mtp{
        .model_path = mtp_model_path,
        .draft_tokens = mtp_draft_tokens,
    };
    return RunRank1Worker(model_path, context, device, gid, bootstrap_host,
                          bootstrap_port, control_port, control_token,
                          worker_max_pending, mtp);
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
  // The model is loaded over the decorator, not the raw peer, so every
  // collective rank 0 issues is counted and its byte size is observable. The
  // scope bind below goes through the same object, and both delegate, so the
  // wire behaviour is exactly the inner communicator's.
  auto collectives = std::make_shared<CollectiveTrace>(std::move(communicator));
  Say("rank0", "RDMA peer established on bootstrap port " +
                   std::to_string(bootstrap_port));
  if (allreduce_bench != 0) {
    return RunAllReduceBench("rank0", *collectives, device, allreduce_bench);
  }

  // `--batched-w2` returns here, BEFORE the control listener binds: this mode
  // has no cohort, no command and no worker, so binding a control peer would
  // only add a second rendezvous that nothing ever connects to. Rank 1 takes
  // the matching branch before its control connect.
  if (batched_w2 || serial_w2) {
    Say("rank0", std::string(serial_w2 ? "serial-w2" : "batched-w2") +
                     " mode: no control wire, no cohort, no worker");
    q::ModelOptions batched_options{
        .max_context = context,
        .mtp_model_path = "",
        .max_draft_tokens = q::kMaxMtpDraftTokens,
        .vision_model_path = "",
        .decode_concurrency = 1,
        .tp_rank = 0,
        .tp_world_size = kWorldSize,
        .hip_device = static_cast<int>(device),
        .communicator = collectives,
    };
    auto moe_inputs = std::make_shared<MoeInputHashes>();
    if (moe_input_hashes) {
      batched_options.moe_observer =
          [moe_inputs](const float* data, std::size_t bytes,
                       hipStream_t stream) {
            moe_inputs->Observe(data, bytes, stream);
          };
    }
    auto batched_model = q::Model::Load(model_path, batched_options, &error);
    if (!batched_model) {
      Warn("batched-w2 model load failed: " + error);
      return kTransportFailure;
    }
    if (batched_model->HasMtp()) {
      Warn("batched-w2 model reports an MTP sidecar, which this mode refuses");
      return kInvalidArguments;
    }
    std::vector<std::int32_t> batched_prompt[kMemberCount];
    for (std::size_t index = 0; index < kMemberCount; ++index) {
      batched_prompt[index] =
          prompt_tokens[index].has_value()
              ? SyntheticPrompt(*batched_model, *prompt_tokens[index], index)
              : batched_model->Tokenize(prompt[index]);
      if (batched_prompt[index].empty()) {
        Warn("batched-w2 member " + std::to_string(index) +
             " prompt tokenized empty");
        return kInvalidArguments;
      }
      if (batched_prompt[index].size() + budget >
          static_cast<std::size_t>(context)) {
        Warn("batched-w2 member " + std::to_string(index) + " prompt has " +
             std::to_string(batched_prompt[index].size()) +
             " tokens, plus --max-tokens " + std::to_string(budget) +
             ", which exceeds the negotiated context " +
             std::to_string(context));
        return kInvalidArguments;
      }
    }
    return RunBatchedW2("rank0", batched_model, batched_prompt, budget,
                         context, collectives, serial_w2,
                         moe_input_hashes ? moe_inputs.get() : nullptr, &error);
  }

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
      // telemetry. With `--expect-worker-mtp` these two fields instead mirror
      // rank 1's loaded sidecar (`has_mtp ? max_draft_tokens : 0` and
      // `use_mtp = has_mtp`, inference_backend.cpp:3841-3846), because
      // `Handshake` compares all eight fields exactly
      // (tp_control.cpp:671-680). Rank 0 still loads no sidecar: it is the AR
      // lockstep driver and this mode asserts that no collective is issued.
      // The handshake field is NOT the same number as the load() argument.
      // A worker reports `has_mtp ? speculative_config.max_draft_tokens : 0`
      // (inference_backend.cpp:3848): it passes 7 to `load()` even without a
      // sidecar, because Model::Load rejects zero, but presents 0 on the
      // wire. Rank 0 must mirror the wire value, not the load value.
      .max_draft_tokens = expect_worker_mtp ? mtp_draft_tokens : 0,
      .use_mtp = expect_worker_mtp,
      .allow_cache_reuse = false,
      .auth_token = control_token,
      .prefill_chunk_tokens = kWorkerPrefillChunkTokens,
  };
  if (!control->Handshake(control_config, &error)) {
    Warn("TP control handshake failed: " + error);
    return kTransportFailure;
  }
  Say("rank0", "TP control handshaken on port " + std::to_string(control_port));
  if (expect_worker_mtp) {
    Say("rank0",
        "FAULT INJECTED: --expect-worker-mtp declares rank 1's handshake, "
        "which presents use_mtp=true and max_draft_tokens=" +
            std::to_string(mtp_draft_tokens) +
            " because that worker loaded a draft sidecar; rank 0 loaded NO "
            "sidecar and keeps the AR lockstep model, and it must issue no "
            "collective at all");
  }

  // 3. Load with the communicator attached so the executor installs its
  //    all-reduce callback; a TP=2 model refuses to load without one.
  q::ModelOptions options{
      .max_context = context,
      // No MTP sidecar and no MTP path, in every mode: rank 0 never loads the
      // draft sidecar, not even when it declares the worker's parity above.
      .mtp_model_path = "",
      .max_draft_tokens = q::kMaxMtpDraftTokens,
      .vision_model_path = "",
      // The C2 slice is serial: one resident request per rank.
      .decode_concurrency = 1,
      .tp_rank = 0,
      .tp_world_size = 2,
      .hip_device = static_cast<int>(device),
      .communicator = collectives,
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
  Say("rank0",
      "model loaded: layers=" + std::to_string(model->config().num_layers) +
          " hidden=" + std::to_string(model->config().hidden_size) +
          " prefill_capacity=" + std::to_string(model->PrefillCapacity()) +
          " context=" + std::to_string(context));

  // 4. Build the two members and reject anything the worker would refuse
  //    before a single collective is issued. `--max-tokens` against `--context`
  //    was already checked before the rendezvous; what is left here needs the
  //    loaded model, because only now does a prompt's token count exist.
  std::vector<std::int32_t> member_prompt[kMemberCount];
  MemberPlan plan[kMemberCount];
  for (std::size_t index = 0; index < kMemberCount; ++index) {
    if (prompt_tokens[index].has_value()) {
      // Exact count by construction, so no tokenizer round trip and no
      // dependency on the vocabulary: the same flag yields the same ids, the
      // same plan digest and the same byte sequence on every run.
      member_prompt[index] =
          SyntheticPrompt(*model, *prompt_tokens[index], index);
    } else {
      member_prompt[index] = model->Tokenize(prompt[index]);
    }
    if (member_prompt[index].empty()) {
      Warn("member " + std::to_string(index) + " prompt tokenized empty");
      return kInvalidArguments;
    }
    // A prompt MAY span chunks. `PrefillCapacity` is what splits it, the worker
    // splits it identically, and `RunMember` reports the byte sequence it
    // actually issued so that claim is checked rather than assumed.
    if (member_prompt[index].size() + budget >
        static_cast<std::size_t>(context)) {
      Warn("member " + std::to_string(index) + " prompt has " +
           std::to_string(member_prompt[index].size()) +
           " tokens, plus --max-tokens " + std::to_string(budget) + " (" +
           std::to_string(member_prompt[index].size() + budget) +
           ") exceeds the negotiated context " + std::to_string(context));
      return kInvalidArguments;
    }
    plan[index] = PlanPrefill(member_prompt[index].size(),
                              model->PrefillCapacity(), model->config());
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
  command.execution_plan_digest = server::ComputeTpExecutionPlanDigest(command);
  command.cache_plan_digest = server::ComputeTpCachePlanDigest(command);
  if (!server::ValidateTpControlCommand(command, &error)) {
    Warn("local C2 command pre-flight failed: " + error);
    return kInvalidArguments;
  }
  Say("rank0",
      "C2 command sequence=" + std::to_string(command.sequence) +
          " cohort=" + std::to_string(command.cohort_id) +
          " execution_plan=" + DigestHex(command.execution_plan_digest) +
          " cache_plan=" + DigestHex(command.cache_plan_digest));

  // 5. `BeginOperation` is local, so it needs no ordering against the worker's
  //    own bind. The scope stays bound across both members: the C2 members
  //    share the command sequence scope, so one lease spans the whole cohort.
  //    `--tp-scope-override` binds a DIFFERENT scope while the command keeps
  //    the normal sequence, so rank 1's worker binds `command.sequence` and the
  //    first `ExchangePartial` header exchange must fail identity validation.
  const std::uint64_t bound_scope = scope_override.value_or(command.sequence);
  OperationScope operation(collectives, bound_scope);
  if (!operation.Begin(&error)) {
    Warn("collective scope bind failed: " + error);
    return kTransportFailure;
  }
  if (!control->SendCommand(command, &error)) {
    Warn("C2 command send failed: " + error);
    return kTransportFailure;
  }
  Say("rank0", "C2 command sent; scope " + std::to_string(bound_scope) +
                   " bound for the whole cohort");
  if (scope_override.has_value()) {
    Say("rank0",
        "FAULT INJECTED: rank 0 bound collective scope " +
            std::to_string(bound_scope) +
            " while the command carries sequence " +
            std::to_string(command.sequence) +
            ", which is the scope rank 1's worker binds; the first "
            "ExchangePartial header exchange must fail with a verbs identity "
            "or size mismatch and poison both communicators");
  }
  if (expect_c2_error) {
    Say("rank0",
        "FAULT INJECTED: --expect-c2-error, so the lockstep members are NOT "
        "executed and no collective is issued by rank 0; only the worker's "
        "response is awaited, and a response without an error is an envelope "
        "mismatch (exit 4)");
  }
  if (expect_worker_mtp) {
    Say("rank0",
        "FAULT INJECTED: rank 1's worker has a draft sidecar loaded, so the "
        "next kCohort2Ar command must be refused BEFORE BeginOperation with "
        "error \"C2 cohort AR is not supported while the MTP sidecar is "
        "loaded\"; rank 1 must bind no lease and issue no collective, and "
        "rank 0 must see that error with BOTH member envelopes within "
        "single-digit milliseconds and exit 7");
  }

  // 6. Lockstep: issue exactly the forwards the worker issues, in order.
  //    `--expect-c2-error` deliberately skips this: the worker refuses the
  //    cohort before it binds a scope -- at admission, or ahead of the bind
  //    when a draft sidecar is loaded -- and never issues a collective, so a
  //    lockstep peer would block on the first prefill exchange and wait out
  //    the 30 s collective timeout instead of reading the C2 error response.
  MemberTrace trace[kMemberCount];
  for (std::size_t index = 0; index < kMemberCount && !expect_c2_error;
       ++index) {
    if (!RunMember("rank0", index, model, plan[index], member_prompt[index],
                   budget, context, *collectives, &trace[index], &error)) {
      Warn("member " + std::to_string(index) +
           " local execution failed: " + error);
      if (scope_override.has_value()) {
        Warn("collective fault detected as designed: rank 0 bound scope " +
             std::to_string(bound_scope) +
             " and rank 1 bound the command sequence " +
             std::to_string(command.sequence) +
             ", so the first ExchangePartial header exchange poisoned both "
             "communicators; this run must fail closed, never pass");
      }
      return kTransportFailure;
    }
    // The hard desync check, BEFORE the member line reports a contract. The
    // prediction and the observation are independent: the plan came from the
    // token count and `PrefillCapacity`, the trace came from the wire. A
    // difference means rank 0's lockstep would not mirror the worker, so the
    // run is a failure in its own right and the member line below, which
    // asserts what the worker must mirror, must not be printed. The two common
    // causes are a `Feed` that no longer chunks at `PrefillCapacity` and a
    // `PrefillCapacity` that is not `exec.max_batch`; the log names the first
    // differing index so either is one glance away.
    if (trace[index].prefill_forwards != plan[index].prefill_forwards ||
        trace[index].prefill_sizes != plan[index].expected_sizes) {
      const std::size_t at = FirstSizeDifference(plan[index].expected_sizes,
                                                 trace[index].prefill_sizes);
      const std::string expected =
          at < plan[index].expected_sizes.size()
              ? std::to_string(plan[index].expected_sizes[at])
              : "end-of-trace";
      const std::string observed =
          at < trace[index].prefill_sizes.size()
              ? std::to_string(trace[index].prefill_sizes[at])
              : "end-of-trace";
      Warn("member " + std::to_string(index) +
           " prefill trace desync: rank 0 issued " +
           std::to_string(trace[index].prefill_forwards) + " forward(s) of " +
           std::to_string(trace[index].prefill_sizes.size()) +
           " collectives [" + ByteRunsText(trace[index].prefill_sizes) +
           "], but the chunking of " +
           std::to_string(plan[index].prompt_tokens) + "t at capacity " +
           std::to_string(plan[index].capacity) + " predicted " +
           std::to_string(plan[index].prefill_forwards) + " forward(s) of " +
           std::to_string(plan[index].expected_sizes.size()) +
           " collectives [" + ByteRunsText(plan[index].expected_sizes) +
           "]; they first differ at collective " + std::to_string(at) +
           " (expected " + expected + "B, observed " + observed +
           "B), so the lockstep would NOT mirror the worker and this run is "
           "not a pass");
      return kCollectiveTraceDesync;
    }
    const std::size_t forwards =
        trace[index].prefill_forwards + trace[index].decode_forwards;
    const char* stop =
        trace[index].stopped_on_token
            ? "token"
            : (trace[index].stopped_on_budget ? "budget" : "none");
    Say("rank0",
        "member " + std::to_string(index) +
            " prompt=" + std::to_string(member_prompt[index].size()) +
            "t tokens=" + std::to_string(trace[index].tokens.size()) +
            " prefill_forwards=" +
            std::to_string(trace[index].prefill_forwards) +
            " decode_forwards=" + std::to_string(trace[index].decode_forwards) +
            " forwards=" + std::to_string(forwards) + " collectives=" +
            std::to_string(forwards * model->config().num_layers) +
            " stop=" + stop + " text=" + model->Decode(trace[index].tokens));
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
  Say("rank0", "C2 response received: members=" +
                   std::to_string(response.members.size()) + " tokens=[" +
                   std::to_string(response.tokens.size()) + "] error=" +
                   (response.error.empty() ? "<empty>" : response.error));

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
         DigestHex(response.cache_plan_digest) +
         " command=" + DigestHex(command.execution_plan_digest) + "/" +
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
           std::to_string(response.members[index].member_id) + ", expected " +
           std::to_string(command.members[index].member_id) +
           " (member order is part of the C2 contract)");
      return kMemberOrderMismatch;
    }
  }
  // A C2 error response is a cohort refusal, not a token disagreement. The
  // stamped banner carries the elapsed time, so an admission refusal (no
  // collective, milliseconds) is distinguishable from a refusal that arrived
  // only after a stalled collective.
  if (!response.error.empty()) {
    const std::string stamp = ElapsedMs();
    std::fprintf(
        stderr, "[c2-probe t=%sms] C2 ERROR RESPONSE for cohort %llu: %s\n",
        stamp.c_str(), static_cast<unsigned long long>(response.cohort_id),
        response.error.c_str());
    std::fflush(stderr);
    // `--expect-worker-mtp` asserts the refusal REASON, not merely that the
    // cohort was refused: every refusal answers with this exit code, so an
    // admission refusal (or a changed production message) must not be able to
    // read as this mode's verdict. The comparison is exact because the worker
    // sends `kMtpRefusal` verbatim and no cleanup failure is appended to a
    // refusal that never bound a scope.
    if (expect_worker_mtp && response.error != kMtpRefusal) {
      Warn(
          "the worker refused the cohort, but not for the injected reason: "
          "expected \"" +
          std::string(kMtpRefusal) + "\", got \"" + response.error +
          "\"; the run still failed closed, so this is not a pass of mode 3");
      return kC2ErrorReasonMismatch;
    }
    return kCohortErrorResponse;
  }
  // `--expect-c2-error` asserts that the worker refused the cohort. Rank 0's
  // lockstep run was deliberately skipped, so there is no local trace to
  // compare and an error-free response is a broken premise, not a pass. The
  // mode can therefore never reach the PASS line below.
  if (expect_c2_error) {
    Warn(
        "--expect-c2-error was set but the worker answered without an error; "
        "rank 0 issued no collective, so this response cannot be verified and "
        "is an envelope mismatch, not a pass");
    return kEnvelopeMismatch;
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
      Say("rank0",
          "member " + std::to_string(index) +
              " AGREES with the local greedy output: " + TokensText(received));
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
    Warn("member " + std::to_string(index) + " token disagreement at index " +
         std::to_string(at) + ": worker=" + worker_token + " local=" +
         local_token + " worker_len=" + std::to_string(received.size()) +
         " local_len=" + std::to_string(local.size()) + " worker_tokens=[" +
         TokensText(received) + "] local_tokens=[" + TokensText(local) + "]");
    return token_mismatch[index];
  }

  Say("rank0",
      "C2 cohort PASS: control plane, ordered two-member execution and "
      "cross-rank token agreement verified");
  return kOk;
}

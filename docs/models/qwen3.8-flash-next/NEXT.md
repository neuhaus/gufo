# Next steps

Forward plan for the Qwen3.8-Flash-Next TP2 and Q8 work, as of 2026-09-25.

Companion documents: [TP2.md](TP2.md) for the C>1 boundary, `Q8.md` (added by
[#1](https://github.com/neuhaus/gufo/pull/1)) for the full-Q8 resume handoff,
[EXPERIMENTS.md](EXPERIMENTS.md) for the evidence log.

## Where things stand

The dormant serial C2 contract is implemented and verified on hardware, and is
split across three reviewable PRs:

| PR | Branch | Content |
|---|---|---|
| [#1](https://github.com/neuhaus/gufo/pull/1) | `pr/q8-ple` | Q8 PLE loader support (independent) |
| [#2](https://github.com/neuhaus/gufo/pull/2) | `pr/tp2-c1` | TP2 RDMA execution and C1 serving |
| [#3](https://github.com/neuhaus/gufo/pull/3) | `pr/c2-dormant` | dormant C2 cohort contract (stacked on #2) |

C2 evidence on hardware: short prompts (3/3 runs), long prompts with chunked
collectives (3000t and 2500t, byte-exact), admission refusal, operation-scope
mismatch on a poisoned communicator, and MTP refusal. Per-member cross-rank
token agreement in every passing run.

The original `feature/qwen38-flash-next-tp2-rdma` branch is retained as a
fallback and is not proposed for merge.

Another branch, `pr/c2-plan-check`, is stacked on #3. It adds the worker's
runner-capacity refusal for C2 (step 1 below) and carries this document. It has
not been built or opened as a PR yet.

## 1. Physical C2 — critical path

Each step depends on the one above it. **Step 1 is a design decision, not a
coding task**, and nothing below it can be specified until it is made.

Recommended order: prototype step 2 first as a width-2 run inside
`qwen38_flash_next_tp_c2_probe`. With no serving or control-wire changes, that
spike does not depend on the step 1 decision. Measure cross-rank token agreement
and throughput against serial C2. Nothing yet shows that C2 is useful, so that
measurement decides whether steps 1 and 3–5 are worth building.

Status: the spike exists as `--batched-w2` in `qwen38_flash_next_tp_c2_probe`.
It builds and passes the format gate, and it is **committed but not yet run on
hardware** — treat it as unverified until a two-host run says otherwise. It
loads the model on both ranks over the engine with the communicator attached,
branches before the control wire on both sides, and runs one identical program:
two live sessions, per-member prefill, then a two-row `EvaluateBatch` per decode
step. Each rank prints its per-member tokens, a checksum, and the exact member
set included in every batched step, so the harness can check **schedule**
agreement as well as token agreement; a member that stops early drops out of
later batches, and a rank disagreement there would diverge the byte counts.
The scope id is a constant both sides bind by construction, so a run is only
comparable when both ranks pass the flag.

The numerics precondition has been measured, on one host, for the one operation
that is actually batched on this path: the single-token advance is token-identical
to serial (`TP2.md`, the single-host batched-advance entry; `plan=batched-w2
batch_width=2` confirmed in the log, `plan=serial-c1` control, all three paths
hashing alike). Batched decode remains unmeasured, and there is no AllReduce in
that test, so the distributed reduction order is still open.

### 1. Carry the execution width — DECISION NEEDED

Correcting an earlier overstatement in this document: `batched-w2` does **not**
lack a valid digest. The digest can be computed for a batched plan. What is
missing is that it **cannot distinguish a serial cohort from a batched one**,
because the execution width is not carried on the wire and the four shape
fields are fixed constants.

`ComputeTpExecutionPlanDigest` hashes the same command bytes on both ranks, so
a digest match proves the command was not mutated. It does **not** prove the two
ranks will execute it the same way. Concretely: if rank 0 assumes a batched
plan and rank 1's runner only supports width 1, the digests still match, the
command is accepted, the operation scope is bound, and the disagreement surfaces
later — as a silent serial fallback (which desynchronises collective byte counts
at the first mismatched collective) or as a throw inside the runner.

Note also that a C2 cohort is **not** a width-2 execution today: the verified
runs show two sequential width-1 executions, each reporting
`physical_execution_width = 1` and `execution_plan = "serial-c1"`. So the width
is not derivable from member count.

Options, in increasing cost:
- **Add a worker-side capability check**, refusing a plan the worker cannot
  honour before `BeginOperation`. Purely local, no wire change, and the digest
  does not change. Checked against `SupportedPlans` it would always pass today,
  because the dormant cohort is serial width 1 and every runner must support
  that plan, so that comparison belongs with the batched runner. The part the
  serial contract does depend on, the worker's runner capacity, is done (below).
- **Add an execution-width field to the command** and have each rank hash it,
  making the digest a genuine pre-flight agreement check. Costs a new field and
  a `kVersion` bump, and it makes a "canonical" digest depend on local
  capability, which the existing doc comment deliberately avoids.
- **Add a distinct command kind** for the batched plan, so the wire format
  itself expresses the plan and the digest follows from the kind.

Apart from the capacity check, all three need a batched runner to be meaningful;
do not build them speculatively. The current shape fields are deliberately left
unnamed, because their per-position meaning was never established and inventing
names would invite exactly the reinterpretation this step is meant to prevent.

Recommended once a batched runner exists: a distinct command kind. The kind is
already hashed into the execution-plan digest, so the digest tells the plans
apart without a new field. If the kind also fixes the cohort's step program
(step 3), both ranks derive the same collectives from the same command.

**Done on `pr/c2-plan-check`: the worker refuses C2 unless its runner capacity
is 1.** The worker loop handles one C1 command at a time, so its capacity never
mattered before. C2 is the first path that puts two requests into the worker's
scheduler at once. With a capacity above 1 the scheduler would admit both
members and advance them in turn, reporting `serial-fallback`, which the
response check accepts, so nothing else stops an interleaved cohort before its
collectives run. The refusal comes before the lease is bound, like the MTP
refusal, and the worker keeps serving. It cannot trigger in production yet,
because TP2 loading requires one session: it guards step 5 and does not fix a
live bug.

### 2. Advertise and implement batched plans for the distributed runner

Correcting an earlier overstatement in this document: **neither entry point
needs a new batched implementation.** Both already have one.

- `AdvanceBatch` below its guard already calls `QwenFlashNextSession::EvaluateBatch`
  (`inference_backend.cpp:2765-2775`), which reaches `EvaluateBatchImpl` and a
  single `ForwardBatch` across all N sessions. The only blocker is the
  `distributed_ && advances.size() > 1` throw at `:2761`.
- `DecodeBatch` reaches `Session::DecodeBatch` -> `RunIsolatedBatch` ->
  `DecodeBatchImpl`, which also ends in one `ForwardBatch` across N sessions.
  "Isolated" refers to per-session attention state, proposal distributions and
  RNG streams staying private, not to serial execution. Its MTP loop breaks
  immediately when a request is not speculative, so **the batched path is not
  MTP-coupled**.

So the distributed work is dispatch, not implementation: advertise `kBatched`
under `distributed_` (`:2542`), lift the two throws (`:2761`, `:2799`), and
settle the `!use_mtp_` fallback at `:2802`, which sends width 2 to the serial
base loop whenever MTP is off.

That last one is the real design question and it is **not** independent of
step 3's MTP decision. A C2 AR cohort refuses MTP permanently by contract
(`tp_cohort_worker.cpp` `kMtpRefusal`, zero draft telemetry per member), yet
the only route to a batched decode runs through the MTP-gated dispatch. Either
a non-MTP width-2 AR decode route is built, or the C2 response contract grows
draft fields. The second option is listed under evidence gaps as an open
question; it is load-bearing here, not a side issue.

**Prefill has no batch form at all.** `Prefill` is per-state
(`text_model_runner.hpp:262`) and neither the runner nor the engine offers a
batched equivalent, so a width-2 program chunks member 0 and then member 1.
Any plan that assumes batched prefill is wrong, including any speedup estimate
that counts it.

### 3. Cohort-level operation scope

One operation lease is bound per control command, scoped to
`command.sequence`, and the communicator pairs a strictly monotonic per-collective
ordinal. The identity part already exists: a `kCohort2Ar` command binds one
lease for both members, and the communicator checks only that both ranks send
the same scope, ordinal and byte count before each collective. A batched step is
one collective with more bytes, so the command's scope can serve as the cohort
identity unchanged.

What is missing is agreement on the collective schedule: which members each
step includes, how prefill is chunked, and when the width drops as a member
finishes. Runner capacity 1 makes that trivial today. Once batched, each rank's
scheduler would decide it independently. A mismatch in bytes or ordinal poisons
both communicators and fails closed. A mismatch that keeps byte counts equal,
such as two members swapped, would sum different members' partials with no
header error.

Recommended: run a cohort as a fixed step program derived from the command,
executed by the same small cohort executor on both ranks instead of the general
scheduler. That means prefill in fixed chunks in member order, lockstep decode
at width 2, then width 1 once a member stops. The only runtime input is the stop
decision, which depends on tokens, so it relies on both ranks computing the same
tokens. The communicator's host sum does not break that, since two operands add
the same in either order. The full-Q8 divergence (section 3) shows it can still
fail, so batched C2 should stay on Q4 until that is understood.

### 4. Scheduler admits a cohort as a batch

Members currently run strictly serially because the worker's runner capacity is
1, which is what makes the dormant path correct. `IsSerialPlanMember` does not
demand it: it also accepts `serial-fallback`, the interleaved order a wider pool
produces. That is why `pr/c2-plan-check` enforces capacity 1 on the worker
before the lease is bound. Real C2 means admitting both members as a batch and
interleaving them, with explicit cache-plan parity.

### 5. Enable `--sessions 2`

Rejected today before communicator and model setup. This is the last step, and
it stays rejected until 1–4 are qualified. Enabling it also trips the worker's
runner-capacity refusal for the serial `kCohort2Ar` kind. That is intended: a
worker running more than one session must not run the serial contract.

## 2. Remaining C2 evidence gaps

- ~~**Peer kill mid-cohort.**~~ **Done, both directions.** Killing the worker
  mid-cohort: the driver failed in 308 ms with `Connection reset by peer` and
  exited non-zero. Killing the driver: the worker failed its member `Wait`, its
  lease release failed on the poisoned communicator, and the worker stopped
  rather than continuing. Both fail closed. Detection is fast because the
  communicator retains the bootstrap socket as the permanent collective channel,
  so a dead peer yields a TCP reset rather than a 30 s collective-timeout stall.
  Not covered: rank 0's response-broker `FailAll` path when the worker dies
  mid-command, and any graceful-shutdown behaviour, since SIGKILL runs no
  cleanup.
- **Member failure.** No natural hardware hook. Would need a fault-injection
  point in the scheduler, which is a production change made for testability
  and should be build-gated rather than always compiled in.
- **Response-broker behaviour when the worker dies.** The peer-kill test killed
  the driver, so the survivor was the worker. The reverse — worker dies, and
  rank 0 must abandon its pending response rather than block in
  `WaitForResponse` — is untested.
- **Invalid-cohort rejection on hardware.** Admission, scope and MTP refusals
  are verified. Rejection of sampled/streaming/cached/continued members is
  covered by hosted tests only; note that the control layer may reject some of
  these before the worker sees them, which is worth determining before writing
  a probe for them. The runner-capacity refusal from `pr/c2-plan-check` is
  hosted-test only: TP2 loading rejects more than one session, so it cannot be
  injected on hardware without relaxing that guard.
- **MTP and C2 together.** Currently refused, permanently by design: the C2
  response contract requires zero draft telemetry per member. Decide whether
  that stays permanent or whether the response contract grows draft fields.

## 3. Full Q8 — blocked

Unqualified and separate from the above. Full Q8 fits memory at ~71,647 MiB GPU
allocation per rank, but long-context AR ranks diverge numerically and the HTTP
path reports `TP worker token mismatch`. Shard hashes are identical on both
hosts, so this is not a transfer problem.

The ordered diagnostic plan is in `Q8.md`, added by
[#1](https://github.com/neuhaus/gufo/pull/1). The short version: a host-only
PLE gather hash for the retained long prompt on both hosts, then a router
global-top-k plan digest to test the leading hypothesis (rank-local routing-plan
desynchronisation at near-ties). Do not change the Q8 quantizer, expert offsets
or reduction order before that evidence exists.

The supported baseline remains Q4 UD-Q4_K_XL plus a shared-Q8 MTP sidecar.

## 4. Verification debt

- **`external-model` and `slow` tests have never been run against these
  branches.** All suite runs used `ctest -E external-model,slow`. That slice is
  unverified and is the largest gap in any "the suite is green" claim.
- **Four Python tests fail on `numpy`**, which the ROCm 7.2.3 dev image does not
  provide. They fail on an unmodified checkout too, so they are not a branch
  defect — but the exclusion is undocumented in CI.
- **`LocalExpert` is currently unused in production.** Only `LocalExpertRange`
  has callers. The `tp_partition` expectation was corrected to match the
  function's documented `global - expert_begin` rule, and that correction is
  safe *only* because nothing consumes `LocalExpert`. If it is ever wired into
  the executor, the reasoning that justified the change no longer holds.
- **`pr/c2-plan-check` has not been compiled or run.** It was written on a host
  with no build environment. Before it becomes a PR it needs the format check,
  `tp_cohort_worker_test` from a `cpu-test` build, and a `gpu-tp2` build for the
  one-line call-site change in `inference_backend.cpp`.
- **Decide the fate of `feature/qwen38-flash-next-tp2-rdma`** (58 commits): keep
  as a fallback or delete once the stack merges.

## Not claimed anywhere in this work

- No batched or physical C2 exists. Everything above concerns the dormant serial
  contract only.
- `--batched-w2` has never been executed. Its presence in the probe is a
  mechanism, not a result: it compiles and is format-clean, and nothing in this
  document should be read as evidence that a batched advance works across two
  ranks until a two-host log says so.
- No performance claim has been measured for C2. Nothing here demonstrates that
  C2 is useful, only that the serial contract is correct.
- Distributed logits are compared with an explicit tolerance, not bit-identity:
  the host float sum changes the reduction order relative to a single device.
  Between the two ranks the sum has two operands, so both ranks get the same
  result and it does not by itself make them disagree.
- Full Q8 is not a supported serving target.

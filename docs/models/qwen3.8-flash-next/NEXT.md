# Next steps

Forward plan for the Qwen3.8-Flash-Next TP2 and Q8 work, as of 2026-09-26.

Companion documents: [TP2.md](TP2.md) for the C>1 boundary, [Q8.md](Q8.md) for
the full-Q8 resume handoff, [EXPERIMENTS.md](EXPERIMENTS.md) for the evidence
log, and [INTEGRATION.md](INTEGRATION.md) for the freshly verified TP2 + Q8
baseline and the rank-0 step-plan implementation handoff.

## Verified integration baseline

`codex/tp2-q8-integration` merges TP2 `d71fc6c` and Q8 PLE `426ab78` at
`0cf273a`. Both hosts built identical release binaries. Eight focused CTests
passed on each host; Q4/Q8 HTTP AR and MTP, repeated output agreement,
multi-chunk prefill, invalid-request refusal and real scope mismatch passed.
Killing rank 1 during Q8 MTP also verified rank 0's C1 response-broker failure
path: HTTP 500 in 0.354 seconds, then immediate refusal on the poisoned
communicator. This closes that C1 evidence gap, not C2 or graceful cancellation.

The integration branch is the baseline for the next coding step. The Q8 and
TP2 PRs stay separate; neither PR branch was rewritten. The rank-0 step plan
has not been implemented. Independent Q8 quality and the full
`slow`/`external-model` suites remain open.

## Where things stand

All branches are rebased onto upstream `main` at `d9a84f1` (2026-09-26), whose
nine new commits add stop sequences, native context defaults, streamed
Responses and cache fixes. The fork's `main` tracks upstream (remote
`upstream` = `gufo-org/gufo`).

| Branch | PR | Content |
|---|---|---|
| `pr/q8-ple` | [#1](https://github.com/neuhaus/gufo/pull/1) | Q8 PLE loader support (independent) |
| `pr/tp2-c1` | [#2](https://github.com/neuhaus/gufo/pull/2) | TP2 RDMA execution and C1 serving |
| `pr/c2-dormant` | [#3](https://github.com/neuhaus/gufo/pull/3) | dormant C2 cohort contract (on #2) |
| `pr/c2-plan-check` | none | worker runner-capacity refusal, this document (on #3) |
| `claude/c2-spike-serial` | none | C2 probe modes and measurements, plus production fixes (on the above) |

`claude/c2-spike-serial` carries production fixes that belong in #2: the
GPU-side add and deferred exchange ack, the idle control-channel keepalive,
cache-prefixed chats run uncached instead of rejected, the Q8 cross-rank fix
and the shared-expert split. Without them #2 on its own ships TP2 with the Q8
divergence and the slower collective. Move them down into #2 and leave only
probe, spike and documentation on top.

Branches to delete: `claude/q8-diag` (a merge made only to diagnose Q8; the fix
is on the tip) and `feature/qwen38-flash-next-tp2-rdma` (the original branch,
replaced by the verified stack). `pr/c2-plan-check` can fold into #3.

Behaviour changed by the rebase: upstream's stop sequences never reach the
rank-1 worker, so TP2 rejects requests with `stop` instead of letting the ranks
stop at different tokens. Upstream's reasoning-token count and context-bounded
token limit are carried into the TP2 request path and the C2 cohort path.

C2 evidence on hardware: short prompts (3/3 runs), long prompts with chunked
collectives (3000t and 2500t, byte-exact), admission refusal, operation-scope
mismatch on a poisoned communicator, and MTP refusal. Per-member cross-rank
token agreement in every passing run.

## Usability and upstream plan

Neither TP2 target is usable in practice yet.

- **Q4 on TP2** works but offers little over one host: 26.9 against 25.9 tok/s
  AR; with MTP (up to seven drafts) 42.03/43.41 tok/s mixed/repetitive against
  one host's 38.87/46.48 on the same requests; and from width 4 up TP2 AR is
  slower (table below).
- **Q8 on TP2** is correct (ranks agree bit for bit; a 2,117-token prompt
  decoded at 23 tok/s over HTTP) and is the reason for TP2, since Q8 does not
  fit one host. Ordinary clients cannot use it yet: requests must be greedy
  (most clients send a nonzero temperature and get a 400), non-streaming, and
  without stop sequences, with no cancellation and one request at a time. Its
  quality has not been checked against a reference.

Most of the request restrictions share one cause: each rank samples its own
tokens and rank 0 compares them at the end. Replacing that with a rank-0 step
plan, where rank 0 samples and sends each step's tokens and stop decisions to
rank 1, enables sampling, stop sequences, cancellation and streaming, removes
the requirement for bit-identical logits that the Q8 bug showed is fragile, and
is the per-step agreement that concurrency needs (section 1, step 3).

**The rank-0 step plan is implemented and verified on two hosts.** Rank 0 samples
and publishes each token as a `kStep` control command; rank 1 consumes it in
`SelectNext` instead of sampling, so its stop test runs on the shared token and
both ranks stop together with no separate stop message. Measured on `fuzzy`/
`misty` at 27.44 tok/s, `serial-c1`, with 64 tokens published and 64 consumed and
byte-identical output to the step-plan-off control — so the mechanism is proven,
not inferred. The per-token exchange costs ~1.9 us one-way (loopback lower
bound), about 0.005% of a 26 tok/s token, so lockstep is not a throughput tax;
see [TP2.md](TP2.md) for the exchange cost, the arming order, and the four
bugs two-host runs found.

What is **not** done is lifting the restrictions the plan unblocks. The
`TP2 requires greedy text without stop sequences` guard still refuses sampling,
stop sequences and streaming.

The rank token comparison fails the request again. It had been demoted to a
logged warning on the reasoning that the token originates on rank 0, but the
step plan covers AR only: with MTP the runner decodes through `DecodeStep`,
which never touches the step channel, so each rank still selects its own
tokens and a divergence would have been served. A difference is never benign:
either a token bypassed the plan or a step was stale or mis-sequenced, and
either way the collectives combined partials from different states. Under the
plan rank 1 also compares its own greedy choice, made on a copy of its sampler,
with each token it consumes, because consuming rank 0's token would otherwise
hide a numerical divergence such as the full-Q8 bug. The first disagreement
fails the request at its end, after both ranks finished the collective schedule
together. Fault injection on two hosts: a forced rank-1 disagreement at step 5
returned HTTP 500 naming the step and both tokens, a corrupted rank-1 token
under MTP returned 500 with the index logged, and in both runs the next request
succeeded with the usual output.

**Stop sequences are a protocol change, not plumbing.** Measured on two hosts:
removing the guard lets a request with `stop` through and it is then **silently
ignored**. A 200-token essay whose text contained the stop word still finished
`length`. The reason is that `TpControlCommand` carries no stop rules at all --
the struct has no stop field -- so rank 1 never receives them and has nothing to
evaluate, while rank 0 has the rules locally. A rank that stops where its peer
does not is a desync at scope teardown, which is exactly where the four bugs
above bit. So the rules have to travel: a field on `TpControlCommand`, a
`kVersion` bump, encoder/decoder/validator changes, and rank 1 passing them into
its own `Submit`. Even then nothing *forces* both ranks to stop on the same
step; identical rules plus identical tokens should make them agree, and the
token comparison is what would show it if they did not.

Rather than shipping the rules, rank 0 can send the decision: it already
evaluates stop sequences, cancellation and client disconnects, and rank 1 always
waits for the next step, so an end marker in place of the next token ends the
request on both ranks at the same step. One mechanism then covers stop
sequences, cancellation and streaming, and nothing depends on both ranks
evaluating the rules identically.

The same design question covers the rest of the remaining work: streaming,
cancellation and a client-side stop all need rank 1 to learn that a request is
over rather than waiting for a token that will not come. That is the thing to
settle before implementing any of them, and it is why they are better done as
one design than three patches.

Order:
1. Open an upstream issue asking whether two-host InfiniBand TP is wanted: it
   adds a libibverbs dependency and hardware upstream likely cannot test, so the
   build gate (`GUFO_ENABLE_TP2_RDMA`) must stay off by default.
2. Move the production fixes from `claude/c2-spike-serial` into #2 and condense
   #2 into a short, reviewable series.
3. ~~Rank-0 step plan.~~ **Done and verified on two hosts** (above), and the
   rank token comparison fails the request again (above). Next: the
   request-end protocol -- an end marker from rank 0 for stop sequences, then
   sampled decoding, then streaming and cancellation, designed together rather
   than one guard at a time; and extending the plan to MTP drafts.
4. Q8 quality against a reference, and the `slow`/`external-model` suites.
5. Submit TP2 C1 and the Q8 PLE loader as two separate upstream PRs, with a
   shared integration qualification. Q8 PLE is independently host-testable;
   full-Q8 GPU serving on the available Strix Halo hosts needs both changes.
6. C2 and higher concurrency later, once it serves concurrent requests; dormant
   code without a caller is hard to justify in review. Park #3 until then.

Transport scope: InfiniBand only until the above is in good shape. RoCEv2 and
RDMA over Thunderbolt/USB4 come after. The communicator's exchange uses
one-sided `IBV_WR_RDMA_READ` and `ibv_query_gid` addressing. RoCEv2 keeps that
model and mainly needs the right GID index. [OdinLink-Five](https://github.com/Geramy/OdinLink-Five)
(Thunderbolt RDMA) also sits under libibverbs, but its provider maps queue pairs
to streams and sends every work request as a two-sided SEND, and it has no
`ibv_query_gid`. It would need a two-sided exchange (post a receive, SEND the
partial) behind the existing `Communicator` interface.

## Measured TP2 performance

Q4 UD-Q4_K_XL, rank 0 `fuzzy`, rank 1 `misty`, FDR InfiniBand, 2026-09-26.
TP2 figures come from `qwen38_flash_next_tp_c2_probe` and are development
measurements, not published cells; single-host figures are from
[BENCHMARKS.md](BENCHMARKS.md). "Now" includes the GPU-side add and deferred
ack described below.

| Workload | One host | TP2 before | TP2 now |
|---|---|---|---|
| Decode, one stream, short context | 26.0 tok/s | 26.5 tok/s | 26.9 tok/s |
| Decode, two streams, short context | 45.7 tok/s | 44.1 tok/s batched | 44.8 tok/s batched |
| Prefill from 0 to 32K tokens | 1,422–1,457 tok/s | 1,070 tok/s | 1,187 tok/s |
| Decode at depth 32K, one stream | 24.3 tok/s | 24.1–24.5 tok/s | 24.9 tok/s |

Batched decode at higher width (`--batched-w2 --width N`, probe default
prompts of 9–13 tokens, 128 tokens per member, greedy). Every width produced
identical tokens on both ranks, and batched matched serial token for token.
Single-host Q4 figures are the HTTP sum of rates from BENCHMARKS.md
(pp2048 prompts), so they are a reference, not a matched control.

| Width | Q8 TP2 step | Q8 TP2 aggregate | Q4 TP2 aggregate | Q4 one host |
|---|---|---|---|---|
| 1 (serial) | 41.3 ms | 24.2 tok/s | 26.9 tok/s | 25.9 tok/s |
| 2 | 50.1 ms | 37.2 tok/s | 44.8 tok/s | 45.7 tok/s |
| 4 | 71.1 ms | 54.4 tok/s | 63.2 tok/s | 76.3 tok/s |
| 8 | 103.3 ms | 76.2 tok/s | 87.7 tok/s | 108.7 tok/s |

Full Q8 does not fit one host, so for Q8 these are the only figures: batching
gives 3.15× at width 8. For Q4, TP2 falls behind one host as the width grows.
The per-layer exchanges take about 10 ms of each step from width 4 up (5 ms at
width 2), although a 4–8 row exchange is only 40–80 KiB; most of it is waiting
for the rank whose experts received more rows. Rank balance, not the fabric,
limits TP2 at higher concurrency.

The upstream headline figures (59.41 tok/s single user, 157.22 tok/s at eight
users) are MTP on the repetitive prompt. On mixed text one-host MTP reaches
106.47 tok/s at eight users, about AR's 108.67, so AR against AR is the matched
comparison above. C2 refuses MTP; the rank-0 step plan would carry draft
lengths and acceptance, which is how C2 can admit MTP.

TP2 C1 serving now runs MTP with the normal adaptive chain of up to seven
drafts; the former one-draft cap had no technical reason. The length controller
depends only on acceptance history, never on timings (`mtp_policy.hpp`), so
ranks with identical tokens choose identical draft lengths, and rank 0 fails a
request whose worker draft telemetry differs. Serving, greedy, 256 tokens after
a ~2,000-token benchmark prompt (`prose`/`repetition` tasks), tok/s:

| Configuration | Mixed | Repetitive |
|---|---|---|
| Q4 TP2, 1 draft | 39.06 | 40.31 |
| Q4 TP2, up to 7 | 42.03 | 43.41 |
| Q4 one host, up to 7 | 38.87 | 46.48 |
| Q8 TP2, 1 draft | 34.93 | 37.70 |
| Q8 TP2, up to 7 | 37.65 | 42.41 |

Both ranks agreed on tokens and draft telemetry in every request, and each TP2
output matched its one-draft output exactly. With MTP, Q8 TP2 decodes 1.6–1.8×
faster than its AR rate (about 24 tok/s).

For Q4, TP2 is a capacity path, not yet a speed path. Profiles and
microbenchmarks show why:

- **Decode is dominated by replicated dense work.** Dense Q8 GEMVs take 57% of
  decode kernel time and routed experts about 13%, and only the routed experts
  are split. Halving them can save about 7% at best.
- **What TP2 adds to a decode token is now about 3 ms:** the exchange itself
  (p50 41 µs inside decode, 48 per token, including waiting for the slower
  rank) and the stream synchronization before each staging copy. Both ranks
  now compute half the shared expert, so the rank-1 wait for rank 0's shared
  expert is gone. The ~4 ms of 2–3 µs gaps between the ~1,780 kernels per token occur on
  one host too.
- **Graphs are not the missing piece.** A temporary single-host build with
  graphs disabled decoded at 26.39±0.55 against 26.73±0.12 tok/s. The TP2 loss
  came from the per-layer synchronization, not from eager launches.
- **Prefill is bound by the NIC's PCIe link.** Both ConnectX-3 cards run at PCIe
  Gen3 x4: `perftest` measures 3.27 GB/s one way and 5.41 GB/s in both
  directions, so a 5 MiB exchange cannot go below about 1.9 ms. The 512-row
  all-reduce now takes 2.9 ms (4.5 ms before), almost all of it the RDMA read.
  At that floor TP2 prefill would reach about 1,310 tok/s, still below one host,
  unless communication overlaps compute.

Proposed order for TP2 speed:
1. Done: the shared expert is split across ranks; each computes half its
   intermediate dimension and the partial sums ride the existing all-reduce.
   It recovered the Q8 fix's ~3% (see `EXPERIMENTS.md`).
2. Split the replicated dense projections (true tensor parallelism for
   attention, GDN and HC projections). This is the only change that can make
   TP2 decode clearly faster than one host, and it adds collectives, so the
   exchange latency matters again.
3. Prefill: overlap the exchange of one half-chunk with the compute of the
   other, since the link itself is at its limit. A Gen4 x4 NIC would halve the
   transfer time.
4. Physical C2 (section 1) is feasible: the spike batched two streams with
   identical tokens and schedules on both ranks, 1.67× over serial on Q4 and
   1.54× on Q8, and widths 4 and 8 also stay in step (table above). On Q4 it
   trails one host from width 4 up; it matters for full Q8, which needs two
   hosts.
5. Higher width: reduce the per-layer wait for the slower rank, for example by
   overlapping the exchange with the next replicated work or by balancing
   expert placement from measured routing.

## 1. Physical C2 — critical path

Each step depends on the one above it. **Step 1 is a design decision, not a
coding task**, and nothing below it can be specified until it is made.

The step-2 spike has run on two hosts. `--batched-w2` loads the model on both
ranks over the engine with the communicator attached, branches before the
control wire on both sides, and runs one identical program: two live sessions,
per-member prefill, then a two-row `EvaluateBatch` per decode step.
`--serial-w2` runs the same program with one member advanced at a time. Each
rank prints its per-member tokens, a checksum and the member set of every step.

Results, Q4, greedy: at 64 and 128 tokens per member with short prose prompts,
and with two 32,000-token prompts, both ranks printed identical tokens,
checksums and per-step member sets, and the batched and serial runs produced
identical tokens. So the two-row advance, including its routed-expert
all-reduce in `MoeBatch`, keeps both ranks on one collective schedule, and it
does not perturb greedy tokens at this scale. Batched decode ran at 45.4 ms per
two-token step against 75.6 ms serial, 1.67× the aggregate throughput. The
numbers are in `EXPERIMENTS.md`; what they mean for priorities is in the
performance section above.

Widths 4 and 8 (`--width N`) also stayed in step on Q4 and full Q8, and Q8
batched matched serial (performance section).

Still open: `DecodeBatch` (as opposed to the advance), sampled decoding, and a
two-stream run at depth. The synthetic 32K prompts stopped the second member
after one token, so that run measured one stream.

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

Earlier recommendation: run a cohort as a fixed step program derived from the
command, identical on both ranks. That only covers members that arrive
together; real traffic arrives staggered, and members join and finish
mid-generation.

Recommended now: rank 0's scheduler owns the schedule and sends a small
per-step plan to rank 1: which members the step includes and in what order,
their tokens, and which members stop. Rank 1 executes the plan and never
samples. This covers any width up to eight and members joining mid-way, and
removes the dependence on both ranks computing the same tokens (the full-Q8
divergence, section 3, showed that dependence is fragile). A digest of the
step's member set in each collective header closes the equal-byte-count hole
above.

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
  Not covered by that cohort probe: rank 0's response-broker `FailAll` path
  when the worker dies mid-command. The integration run now covers C1 (below).
  Graceful-shutdown behaviour remains open, since SIGKILL runs no cleanup.
- **Member failure.** No natural hardware hook. Would need a fault-injection
  point in the scheduler, which is a production change made for testability
  and should be build-gated rather than always compiled in.
- **Response-broker behaviour when the worker dies.** C1 is now verified by
  the integration run: killing the Q8 MTP worker made rank 0 leave its pending
  response wait and return HTTP 500 in 0.354 seconds. The historical C2 probe
  did not exercise the rank-0 serving broker; that C2 case remains open.
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

## 3. Full Q8 — cross-rank divergence fixed

Full Q8 fits memory at ~71,647 MiB GPU allocation per rank. The long-context
divergence and the HTTP `TP worker token mismatch` had one root cause: only rank
0 ran the shared expert, and its Q8_0 projections re-key the executor's
activation staging caches. Rank 1 then reused a differently rounded staged copy
in later projections. Every rank now runs the shared expert and peers drop its
output (see `EXPERIMENTS.md`). The Q8 PLE gather was excluded first (#1).

Q8 ranks now agree bit for bit on the retained long prompt, and Q8 TP2 serving
answered a 2,117-token prompt at 23.0 tok/s decode. The shared-expert split
recovered the ~3% the fix cost. Batched Q8 decode stays in step at widths 2, 4
and 8 and reaches 76.2 tok/s aggregate at width 8.

Still open before Q8 is a supported target: a quality check against a reference
(Q8 does not fit one host, so it needs a CPU or reference-logit comparison),
the `slow` and `external-model` suites, and the request restrictions listed
under "Usability and upstream plan". The integration branch's `Q8.md` records
the cross-rank fix and fresh C1 qualification; #1 remains an independent PR.

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
- **Format checks before 2026-09-26 afternoon were invalid.** The `nixbox`
  container on `fuzzy` still mounted the repository directory from before it
  was re-cloned, so it checked stale code and passed. After the rebase every PR
  head was checked on a fresh `git archive` export and passes; the fixes are one
  folded resolution line and two `style(tp2)` commits. Check on an export, or
  restart `nixbox`, until its mount is current.
- **After the rebase** every PR head builds on its own (misty), the tip builds
  on both hosts, the serving, scheduler and TP tests pass, and Q4 TP2 serving
  answered correctly on both hosts, rejected a `stop` request with 400 and kept
  serving. `inference_backend_gpu_test` skipped for lack of a model, which is
  not a pass.
- **`pr/c2-plan-check`**: the capacity refusal stays hosted-test only.

## Not claimed anywhere in this work

- No physical C2 exists in serving. The batched advance has run only inside the
  probe, with greedy decoding and no MTP.
- C2 throughput is measured only in the probe, at short context and with one
  run per configuration. Nothing is measured through serving.
- Distributed logits are compared with an explicit tolerance, not bit-identity:
  the host float sum changes the reduction order relative to a single device.
  Between the two ranks the sum has two operands, so both ranks get the same
  result and it does not by itself make them disagree.
- Full Q8 is not yet a supported serving target: the ranks now agree, but its
  quality has not been checked against a reference, and TP2 serving accepts
  only greedy, non-streaming requests without stop sequences.
- Neither TP2 target is claimed to be usable with ordinary clients yet.

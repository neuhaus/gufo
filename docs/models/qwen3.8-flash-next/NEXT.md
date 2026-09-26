# TP2 and full Q8: status and plan

As of 2026-09-26. This is the only status document for the fork's TP2 and
full-Q8 work: update it when the status changes and do not restate status
elsewhere. [TP2.md](TP2.md) is the reference (design, running, qualifying);
[EXPERIMENTS.md](EXPERIMENTS.md#tp2-and-full-q8) holds the evidence, one row per
experiment.

## Branches

The fork's `main` tracks upstream `gufo-org/gufo` at `d9a84f1`.

| Branch | PR | Content |
|---|---|---|
| `pr/q8-ple` | [#1](https://github.com/neuhaus/gufo/pull/1) | Q8 PLE loader (independent) |
| `pr/tp2-c1` | [#2](https://github.com/neuhaus/gufo/pull/2) | TP2 RDMA execution and C1 serving |
| `pr/c2-dormant` | [#3](https://github.com/neuhaus/gufo/pull/3) | dormant C2 cohort contract and capacity refusal (on #2) |
| working branch | none | everything above plus the work listed below |

The working branch carries production changes that belong in #2 and are not in
it: the faster exchange (GPU-side add, deferred ack), the idle control-channel
keepalive, cache-prefixed chats run uncached, the Q8 cross-rank fix, the
shared-expert split, MTP with up to seven drafts, the rank-0 step plan and the
rank agreement gate. It also merges the Q8 PLE loader from #1 and carries the
C2 probe, measurements and documentation.

## What works

Two-host serving (`fuzzy` rank 0, `misty` rank 1, FDR InfiniBand) of one
request at a time, Q4 and full Q8, AR and MTP, with sampling, streaming, stop
sequences and client cancellation. Both hosts stay bit-identical; lost peers,
scope mismatches and rank disagreement fail the request with HTTP 500. Full Q8
needs both hosts; it does not fit one.

| Decode, tok/s | Q4 TP2 | Q4 one host | Q8 TP2 |
|---|---|---|---|
| AR, one stream | 26.9 | 25.9 | 24.2 |
| MTP greedy, mixed / repetitive | 41.7 / 43.0 | 38.7 / 46.4 | 37.4 / 42.3 |
| MTP sampled (T 0.7, top-p 0.95), mixed / repetitive | 35.5 / 38.5 | 33.1 / 34.4 | |
| AR batched, width 2 / 4 / 8 (probe only) | 44.8 / 63.2 / 87.7 | 45.7 / 76.3 / 108.7 | 37.2 / 54.4 / 76.2 |

Prefill from 0 to 32K runs at 1,187 tok/s on TP2 against 1,422–1,457 on one
host. Single-host batched figures are HTTP sums at 2K prompts, so that row is a
reference rather than a matched control.

Why TP2 is not faster than one host on Q4 (profile of 2026-09-26, both ranks):

- Only routed experts are split, about 13% of decode kernel time. Q8 GEMVs are
  56.5% and replicated: GDN input projections 22.7%, projections back to the
  hidden size 12.3%, LM head 9.6%, attention QKV 6.3%; hyper-connection
  projections add about 17%.
- TP2 adds about 3.9 ms per token of exchange gaps (48 exchanges): a median
  exchange of about 45 µs, plus waiting for the other rank, whose routed
  experts take on average 38 µs longer or shorter per layer (about 1.8 ms per
  token), because whole experts are assigned to one rank.
- Prefill is capped by the NICs' PCIe Gen3 x4 links (3.27 GB/s one way).
- At width 8 the exchange gaps are about 8.5 ms per step (177 µs mean per
  layer). The exchange itself grows to about 115 µs at 80 KiB (95 µs isolated in
  the microbenchmark) through its serialized steps: stream wait, staging copy,
  TCP header, RDMA read, GPU add. Expert imbalance stays about 35 µs per layer.

What would help, in order of payoff for effort:

1. Split the LM head by vocabulary: about 1.9 ms per token, and only a small
   candidate exchange, since rank 0 already chooses the token.
2. Split every routed expert's intermediate dimension across the ranks, as the
   shared expert already is, instead of assigning whole experts: removes the
   imbalance at every width without adding exchanges.
3. Cheapen the exchange: have the MoE epilogue write the partial straight into
   the RDMA-registered window (Strix Halo's memory is unified), and replace
   the TCP header with an InfiniBand send with immediate data.
4. Split the GDN and attention projections by heads, with the output
   projections split by input: roughly 6–7 ms per token less work for a second
   exchange per layer, so it pays most after 3.

## What does not work yet

- **Upstream sampling defaults.** Upstream's pending
  `feat/model-sampling-defaults` (#277) makes requests without an explicit
  temperature sampled (1.0 / top_p 0.95 / top_k 20); TP2 now handles those
  with MTP.
- **Cache reuse.** Every request prefills its whole prompt, so multi-turn chats
  re-prefill their history; `--tp-cache-reuse` is refused until snapshots are
  mirrored.
- **Q8 quality.** The ranks agree with each other, but full Q8 has not been
  compared with a reference. It needs a CPU or reference-logit comparison,
  since Q8 does not fit one host.
- **Concurrency.** One request at a time; C2 is dormant.

## Current work: rank 1 as an executor

Decided 2026-09-26: replace the step plan and rank 1's mirrored scheduler with
an executor. Rank 0's scheduler stays unchanged; a runner wrapper on rank 0
sends each model call to rank 1 as an instruction just before running it, and
rank 1 runs the same call on the same state. Both ranks then make identical
model calls in identical order by construction, and a request ends on rank 1
when rank 0 says so. This replaces plan items 1–3 below.

Progress (update after every step):

- [x] Audit of the model calls (below).
- [x] Protocol: instruction frame, protocol v7, hosted tests.
- [x] Rank-0 runner wrapper and rank-1 executor loop; rank 1 builds no pool.
- [x] Request paths: sampling, streaming, stop sequences and cancellation no
      longer refused; both entry points share `StartTpRequest`.
- [x] Greedy MTP through a `decode` instruction.
- [x] Two-host qualification (TP2.md checklist; see EXPERIMENTS.md, "TP2
      rank-1 executor"). Speed unchanged against the pre-executor binary.
- [x] Sampled MTP: the `decode` instruction carries rank 0's draw state and
      the begin command the sampling configuration (protocol v8).
- [ ] Cache reuse (mirrored snapshots; outline below), then C2 on the
      batched calls.

Status: qualified on two hosts for Q4 and full Q8, AR and MTP. Sampling,
streaming, stop sequences and client cancellation work, and sampled requests
use MTP. Next in this line: cache reuse, then C2.

**Audit.** The scheduler reaches the model only through `TextRunnerPool`, which
calls the runner and the request states:

| Call | Model work | Instruction |
| --- | --- | --- |
| `Prefill(state, prompt, offset, max)` | chunk forward via `Session::Sync`, exchanges | `prefill(state, offset, max, prompt_size)` |
| `Advance(state, token)` | one forward, exchanges | `advance(state, token)` |
| `DecodeStep(state, max, sampler)`, MTP | draft and verify, exchanges | `decode(state, max)`, greedy only at first |
| `state.Invalidate()` (called by the cache, not the runner) | session reset | `invalidate(state)` |
| `PreparePrefixReuse` | resets the MTP draft-length controller | later, with cache reuse |
| `Snapshot`, `RestoreOrFork` | host copies of session state | later, with cache reuse |
| `AdvanceBatch`, `DecodeBatch` | batched forwards | later, C2 |
| `SelectNext`, `PreviewFirstToken`, `CheckpointPosition`, `Decode` | logits and bookkeeping only | none |

Traps the audit found:

- `Session::Sync` keeps the longest common prefix with the session's contents,
  so a missed reset on rank 1 changes how much it prefills. Every state reset
  must travel, including the ones the continuation cache makes directly.
- The base `TextModelRunner::DecodeStep` (used for one-token budgets and
  without MTP) calls `SelectNext` and `Advance` on the inner runner, so a
  forwarded `DecodeStep` would bypass the wrapper. The wrapper implements that
  path itself: select locally, then a mirrored `advance`.
- The session's cancellation check can abort between layers, mid-forward, which
  would strand rank 1 inside an exchange. The wrapper never forwards it;
  cancellation acts only between calls (the scheduler already checks there),
  so a long prefill is cancelled at the next chunk.
- Rank 1 must not build a pool: it would allocate a second session. It creates
  the same number of states (one) and executes on those.

**Design.**

- A request starts with the existing `kSingle` command, which now also carries
  whether sampling is greedy. Rank 1 binds the operation lease and runs
  instructions until `end`.
- Instruction frame: request sequence, a channel-wide monotonic index, the
  operation, the state id and its arguments. A state reset is valid between
  requests; a forward-bearing instruction outside a request is a protocol error.
- Rank 0 sends each instruction before running the call, so rank 1 starts in
  parallel, then records the call's result in a running digest. Rank 1 does the
  same. `end` carries rank 0's digest and instruction count; any difference
  fails the request with rank 1's error in the response.
- Under greedy decoding rank 1 also compares its own argmax with each `advance`
  token, which keeps the check that caught the full-Q8 bug.
- A call that fails deterministically fails identically on both ranks and is
  recorded in both digests. A rank-0 failure inside a forward leaves rank 1 in
  an exchange until the collective timeout, and the communicator then fails
  closed.
- Sampled requests with MTP loaded decode token by token (local selection plus
  `advance`) until sampled MTP is mirrored.
- `--tp-cache-reuse` is refused in executor mode until snapshots are mirrored.
- The dormant C2 worker path needs rank 1's scheduler, so it is retired; C2
  returns later on batched instructions.
- Cost: one small message per model call, as with the step plan.

## Next: cache reuse on the executor

Not started. Without it every TP2 request prefills its whole prompt, so a
multi-turn chat re-prefills its history on every turn (about 1,190 tok/s on
TP2, so roughly 7 s per 8K tokens of history), and `--tp-cache-reuse` is
refused. The step-plan design had qualified symmetric rank-local reuse
(EXPERIMENTS.md, "TP2 symmetric live-prefix handoff" and "TP2 rank-local
immutable snapshot boundary") because rank 1 ran its own identical scheduler
and cache. Rank 1 now has neither, so every cache action that touches model
state must reach it as an instruction, like the prefill and decode calls do.

### What the cache does on one host

`TextRunnerPool` owns a `ContinuationCache` of request states plus immutable
host snapshots, and on a request:

1. **Acquire** picks a state: a *live hit* keeps a state whose session already
   holds a prefix of the prompt (no copy); a *snapshot hit* restores an older
   boundary into a state (`RestoreOrFork`); otherwise it resets a state. A hit
   calls `PreparePrefixReuse` (Flash-Next: checks the position and resets the
   MTP draft-length controller).
2. **Prompt snapshot**: after prefilling up to the stable prompt boundary
   (`cache_prefix_tokens`, before the assistant framing), the pool captures a
   snapshot on a `std::async` worker (`runner->Snapshot`) while the state is
   frozen, and joins the capture before the next call that mutates the state.
   A failed capture is skipped, not fatal.
3. **Commit** records the prompt snapshot and the live frontier as reusable;
   eviction drops snapshots under a host-memory budget
   (`HostSnapshotBudgetBytes`, half of available memory).

On Flash-Next a snapshot is a host copy of the session (KV cache, recurrent
state, draft-block state, last logits), about 120–170 MB per resident session
at the depths measured. Both ranks hold identical session contents, so each
rank can keep its own copy; snapshot bytes never cross hosts.

### What must be mirrored

| Rank-0 call | Instruction | Rank 1 |
| --- | --- | --- |
| `Snapshot(state)` (capture worker) | `snapshot(state, id)` | capture its own copy into a table under `id` |
| destruction of a mirrored snapshot | `drop(id)` (also between requests) | free the table entry |
| `RestoreOrFork(state, snapshot)` | `restore(state, id)` | restore from its table |
| `PreparePrefixReuse(state, prefix)` | `reuse(state, prefix_size)` | the same call with the `kSingle` prompt's first `prefix_size` tokens |
| `PrepareCancellation(state)` | `cancel-prepare(state)` | the same call (a no-op on Flash-Next; mirrored for other runners) |
| `SnapshotPayloadBytes`, `CheckpointPosition` | none | read-only |

Live hits need nothing extra: every call that changed the state was already
mirrored, so rank 1's session holds the same prefix.

### Design

1. **Snapshot identity and lifetime.** `TpMirroredRunner::Snapshot` returns a
   `TpMirroredSnapshot` (a `TextRunnerSnapshot` holding the inner snapshot and a
   channel-wide monotonic id, never reused) and sends `snapshot(state, id)`
   *before* copying. Its destructor sends `drop(id)`; destruction can happen
   on the scheduler thread, in the cache or at shutdown, so the send is
   `noexcept` and a failure is recorded like a failed reset. `PayloadBytes`
   forwards to the inner snapshot, so rank 0's budget accounting is
   unchanged. If rank 0's own capture throws after the instruction went out,
   the wrapper sends `drop(id)` before rethrowing so rank 1 does not keep an
   orphan.
2. **Ordering.** The capture runs on the pool's worker thread, not the
   scheduler thread. That is safe because the pool freezes the state: no call
   that mutates it is made (so no instruction for it is sent) until the
   capture is joined, and the capture's instruction goes out before the copy
   starts. The channel send is already serialized by the sink's mutex. Add an
   assertion that `Snapshot` is only called while a request is being mirrored,
   and a hosted test that holds the capture open while the scheduler tries to
   proceed.
3. **Rank-1 capture.** First version: synchronous when the instruction
   arrives. Rank 0 cannot advance that state until its own capture is joined,
   so rank 1's copy overlaps rank 0's and the stall is roughly the same copy
   time on both. Measure it; only if rank 1 becomes the laggard, capture on a
   rank-1 worker and join before the next instruction that names the state.
4. **Digest.** Add the snapshot id, and after `restore` and `reuse` the state's
   `CheckpointPosition`, so a missing or mismatched restore fails the request.
5. **Asymmetric failure.** A rank-1 capture or restore can fail where rank
   0's succeeded (for example rank 1 has less free host memory). Then rank 0's
   cache holds an entry rank 1 cannot restore, and every later hit on it would
   fail. Recovery, in order of preference:
   - rank 1 lists the ids it failed to capture or no longer holds in its
     `kEnd` response, and rank 0 invalidates the cache entries that hold them
     (needs a `TextRunnerPool`/`ContinuationCache` call to drop entries by
     snapshot);
   - simpler fallback: on any rank-1 failure in a cached request, rank 0
     clears its whole continuation cache, which resets and drops everything on
     both ranks.
   Either way the failing request itself returns HTTP 500, and the next one
   succeeds without reuse.
6. **Memory.** Rank 1 allocates the same snapshots as rank 0 but is not
   consulted by rank 0's budget. Take the budget as the smaller of the two
   hosts' at startup: rank 1 can report its `HostSnapshotBudgetBytes` in the
   handshake. This also bounds the orphan risk.
7. **Configuration.** Remove the `--tp-cache-reuse` refusal in `load()` and the
   snapshot check in the `TpMirroredRunner` constructor; forward the inner
   runner's `snapshot`, `fork`, `prefix_reuse` and `preserve_snapshot_prefix`
   capabilities and `retained_snapshot_capacity_bytes` instead of clearing
   them. Keep the handshake requiring both ranks to agree on the flag. Disk
   persistence stays refused: snapshots are rank-local and never serialized.
8. **Protocol v9**: instruction operations `snapshot`, `restore`, `drop`,
   `reuse`, `cancel-prepare`, and a snapshot id field; `drop`, like a reset,
   may arrive between requests.

### Implementation order

1. Protocol and validation, with round-trip and refusal tests in
   `tp_control_test`.
2. `TpMirroredSnapshot` and the wrapper calls; the rank-1 snapshot table and
   executor cases; digest additions.
3. Extend `tp_executor_test`: give the toy runner snapshots (copy its tokens),
   send `cache_prompt` requests, and check that both ranks make identical
   calls across a multi-turn conversation (live hit), a branch back to an
   older boundary (snapshot restore), eviction (every rank-0 drop reaches rank
   1, and rank 1's table size equals the snapshots rank 0 still holds), a
   capture on a worker thread, and a failed rank-1 capture (the request fails,
   the cache recovers, the next request succeeds). Mutation-check each: a
   missing `drop`, a missing `reuse`, a restore of the wrong id.
4. Failure recovery (item 5), then the memory budget exchange (item 6).
5. Two-host qualification.

### Qualification on two hosts

- Multi-turn greedy chat with `cache_prompt`: turn two reports
  `cached_prompt_tokens` close to the previous turn's length, prefill time
  drops accordingly, and the output equals the same conversation served
  uncached.
- Branching: two continuations of one prefix restore the older boundary, and
  both outputs equal their uncached versions.
- Sampled and MTP requests with reuse (the draft-length controller is reset on
  reuse on both ranks).
- Eviction under a small budget: rank 1's resident snapshot bytes track rank
  0's, and nothing leaks after many requests.
- Client disconnect during a prompt-snapshot capture, then a cached request.
- Killing rank 1 fails closed as before.
- Speed: time to first token for a turn with 8K tokens of reusable history,
  against uncached TP2 and against one host with its cache.

### Qualification in progress: not finished

Run on `fuzzy`/`misty` and on `fuzzy` alone, Q4 UD-Q4_K_XL, greedy, protocol
v9, binary `1bd3cfbc…` on both hosts. **The qualification is not complete** and
the criterion "the output equals the same conversation served uncached" is
**not** what the code currently satisfies on TP2.

**Equal where it matters.** A four-turn conversation, greedy, produced
byte-identical output in all three configurations -- one host, one host with
`--cache-disk`, and TP2 with `--tp-cache-reuse`:

| turn | one host | one host + disk | TP2 |
| --- | --- | --- | --- |
| 1 | `1c650a7f…` | `1c650a7f…` | `1c650a7f…` |
| follow-up 1 (86 cached / 27 prefill) | `1c650a7f…` | `1c650a7f…` | `1c650a7f…` |
| follow-up 2 (176 / 24) | `ba19e9c3…` | `ba19e9c3…` | `ba19e9c3…` |
| follow-up 3 (201 / 23) | `ec7d56a0…` | `ec7d56a0…` | `ec7d56a0…` |

Cached equalled uncached on every turn on both paths, and the live-frontier
reuse path is identical on both: 86, then 176, then 201 cached tokens with
23--27 prefilled. TP2 produced no divergence warning and no rank-1 error, and
the end-of-request digest did not fail.

**The restore path is not equivalent, and that is the finding.** Replaying an
*identical* request is the one shape that selects the snapshot rather than the
live frontier, because the live frontier is then longer than the prompt:

| | one host | TP2 |
| --- | --- | --- |
| cached tokens | 23 (the whole prompt) | 16 |
| prefilled | 0 | 7 |
| restore | 4.2 ms | 93.1 ms |

The cause is `inference_backend.cpp`, the `allow_distributed_snapshots_ &&
request.cache_prompt` block that re-renders the conversation with
`add_generation_prompt = false` and sets `cache_prefix_tokens` to the common
prefix with that stripped render. It is TP2-only, so on TP2 the checkpoint is
the prompt minus its generation prompt, and the generation prompt is
re-prefilled on every hit. Two consequences:

- The hit is systematically smaller than the prompt by the length of the
  generation prompt. A reported "55 of 78 kept" is this, not a defect in the
  matching: `common_prefix_tokens` in the request log is exactly the stripped
  length.
- Output is still byte-identical, so this is a cache-behaviour and cost
  difference, not a correctness one. But it means the two paths do not have the
  same cache semantics, and a cross-path comparison of `cached_tokens` is
  meaningless.

TP2 restores **fewer** tokens in **22x** the time because the mirrored
`kRestore` waits for rank 1 to restore as well; that 93.1 ms is the dominant
cost of a hit.

**Reuse currently costs more than it saves at these sizes.** On a 200-token
prompt the cached TTFT was 463 ms against 243 ms for a full prefill, and
`prefill_tps` was 58--88 on hits against 459--709 on misses: restoring a
~120 MB snapshot costs more than the 23-token prefill it avoids. This is the
speed criterion above, and it currently fails at small prompt sizes.

**Not reproduced: a cached/uncached difference.** The difference that prompted
this qualification could not be reproduced in any shape tried -- multi-turn
append, agent-style discard, branch, identical replay, with and without
`--cache-disk`, one host and TP2. Cached and uncached were byte-identical
everywhere.

**Unresolved: the output is not invariant across server lifetimes.** The same
request on the same binary produced two different *stable* hashes during one
session, `1c650a7f…` early and `e6fb9c8d…` later, on the one-host path *and*
on TP2, each perfectly reproducible within a server's lifetime (4/4, 6/6, 9/9).
It is not explained by cached versus uncached. It did not reproduce under six
rounds of the target request alone, nor under six rounds interleaved with
cached churn. This is a better candidate for the original "different in
responses" observation than the cache is, and it needs its own bisect: alternate
the two paths on an identically prepared GPU state and see whether the hash
tracks allocator or residency state.

Still to qualify: branching to an older boundary, eviction under a small
budget, sampled and MTP requests with reuse, disconnect during capture, and the
8K-history speed case.

**Environment.** The two ranks need the host to themselves. A leftover server
container left rank 0 with 36 GiB free and it failed with `hipMalloc failed for
stacked tensor`, which rank 1 reports as a misleading handshake error.

## Plan

1. **Executor** (current work, above). It covers what the earlier plan split
   into an end marker, sampled AR and a step plan for MTP: rank 0's scheduler
   decides stops, cancellations and disconnects, and rank 1 simply receives no
   further model calls. MTP decisions stay inside `Session::DecodeStep`, so
   greedy MTP runs the same call on both ranks and compares results; sampled
   MTP needs rank 1's sampler to match rank 0's (RNG state, pending draw and
   accepted history).
2. **Q8 quality** against a reference, and the `slow`/`external-model` suites.
3. **Upstream.** Open an issue asking whether two-host InfiniBand TP is wanted:
   it adds a libibverbs dependency and hardware upstream likely cannot test,
   and the build gate must stay off by default. Then move the working-branch
   changes into #2, condense it into a short series, and submit TP2 C1 and the
   Q8 PLE loader as separate PRs with a shared qualification.
4. **Speed.** Split the dense projections (the only change that can make TP2
   decode clearly faster than one host; it adds collectives), overlap prefill
   exchanges with compute, and reduce the rank wait at higher width by
   overlapping or by balancing expert placement from measured routing.
5. **C2 and concurrency**, once the above works. Park #3 until C2 has a caller.

Transport: InfiniBand only until this plan is done. RoCEv2 keeps the one-sided
RDMA-read design and mainly needs the right GID index.
[OdinLink-Five](https://github.com/Geramy/OdinLink-Five) (Thunderbolt RDMA)
also sits under libibverbs, but its provider sends every work request as a
two-sided SEND and has no `ibv_query_gid`, so it needs a two-sided exchange
behind the `Communicator` interface.

## C2 decisions still open

- **Per-step schedule.** Extend the rank-0 step plan with each step's member set
  and order, so rank 1 never schedules on its own; members may join and finish
  mid-generation. Add a digest of the member set to each exchange header: equal
  byte counts do not detect two swapped members.
- **Execution width on the wire.** The plan digest proves the command was not
  mutated, not that both ranks execute it at the same width. Once a batched
  runner exists, add a distinct command kind for the batched plan; the kind is
  already hashed into the digest.
- **Batched dispatch.** `AdvanceBatch` and `DecodeBatch` already reach one
  `ForwardBatch`; the distributed runner blocks them with two throws, and a
  width-2 AR decode falls back to the serial loop when MTP is off. Either build
  a non-MTP batched decode route or let the C2 response carry draft telemetry.
- **Prefill** has no batch form; members prefill one after the other.
- **`--sessions N`** is enabled last; it also trips the worker's capacity
  refusal for the serial C2 contract, as intended.
- **Evidence gaps:** member failure on hardware (needs a build-gated fault
  hook), rejection of sampled/streaming/cached members on hardware, and rank
  0's broker when the worker dies during a C2 command.

## Verification debt

- The `slow` and `external-model` suites have never run on these branches.
- Four Python tests fail without `numpy`, which the ROCm 7.2.3 dev image lacks;
  they also fail on an unmodified checkout.
- `LocalExpert` has no production caller. Its test expectation was corrected to
  the documented `global - expert_begin` rule; revisit if it gets one.
- The `nixbox` container on `fuzzy` mounts a stale checkout; format checks run
  there before 2026-09-26 afternoon checked old code. Check on a `git archive`
  export.
- `inference_backend_gpu_test` skips without a model; a skip is not a pass.
- The step message's cross-host cost is measured only end to end (26.69 tok/s
  median of ten against a single 27.3 control); its loopback cost is 1.9 µs.

## Not claimed

- No C2 in serving; batched decode has run only in the probe, greedy, no MTP,
  one run per configuration.
- TP2 and one-host logits differ by reduction order and are compared with a
  tolerance, not bit-identity.
- Full Q8 is not a supported serving target.
- TP2 serves one request at a time.

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
sequences, client cancellation and history reuse as on one host.
Both hosts stay bit-identical; lost peers, scope mismatches and rank
disagreement fail the request with HTTP 500. Full Q8 needs both hosts; it does
not fit one.

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
- [x] Cache reuse: mirrored snapshots, prefix reuse and drops (protocol v9),
      qualified on Q4 (below).
- [ ] C2 on the batched calls.

Status: qualified on two hosts for Q4 and full Q8, AR and MTP. Sampling,
streaming, stop sequences and client cancellation work, and sampled requests
use MTP, and history is reused as on one host. Next in this
line: C2.

**Audit.** The scheduler reaches the model only through `TextRunnerPool`, which
calls the runner and the request states:

| Call | Model work | Instruction |
| --- | --- | --- |
| `Prefill(state, prompt, offset, max)` | chunk forward via `Session::Sync`, exchanges | `prefill(state, offset, max, prompt_size)` |
| `Advance(state, token)` | one forward, exchanges | `advance(state, token)` |
| `DecodeStep(state, max, sampler)`, MTP | draft and verify, exchanges | `decode(state, max)`, greedy only at first |
| `state.Invalidate()` (called by the cache, not the runner) | session reset | `invalidate(state)` |
| `PreparePrefixReuse` | resets the MTP draft-length controller | `reuse(state, prefix_size)` |
| `Snapshot`, `RestoreOrFork` | host copies of session state | `snapshot(state, id)`, `restore(state, id)`; `drop(id)` when rank 0 frees it |
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
- The dormant C2 worker path needs rank 1's scheduler, so it is retired; C2
  returns later on batched instructions.
- Cost: one small message per model call, as with the step plan.

## Cache reuse on the executor

Done 2026-09-26 and the default (protocol v10; `--tp-cache-reuse` is gone);
design in [TP2.md](TP2.md#cache-reuse), evidence in EXPERIMENTS.md ("TP2 cache
reuse on the executor").

How it differs from the outline: rank 1 acknowledges every cache operation
before either rank continues, so an asymmetric failure is known at once
instead of reported in `kEnd`; a failed capture on either rank skips the
snapshot (as on one host) and rank 0 drops the id; any other rank-1 failure
fails the request and rank 0 clears its whole cache. The first hardware run
found TP2 caching differently from one host (a replay restored 16 of 23 tokens
in 93 ms): two workarounds from rank-local caching, a stable-prefix boundary
without the generation prompt and `preserve_snapshot_prefix`, were still
active. Both are removed, and the control socket now sets `TCP_NODELAY`.

Qualified on Q4: TP2 cached output equals one-host cached output with the same
hits; a replay restores the whole prompt in about 5 ms; 5.9K tokens of history
reach the first token in 124–141 ms instead of about 4.9 s; MTP speed is
unchanged; functional and kill checks pass; rank 1's memory stays flat.

Open:

- **The qualification criterion was wrong.** "Cached output equals uncached"
  does not hold on one host either: prefilling a short suffix on a cached
  prefix runs different kernel shapes than prefilling the whole prompt, and a
  close greedy choice can flip (seen on turns 3 and 4 of the test chat, AR and
  MTP). Compare TP2 with one host under the same cache hits instead.
- **Branching** to a point inside an earlier turn misses on both paths: the
  prompt snapshot sits after the generation prompt, which a later render of
  that turn does not contain. It is a one-host property, now shared by TP2.
- **Cross-lifetime hashes.** The first hardware run saw one request give two
  stable hashes across server lifetimes. Seven server lifetimes today (one
  host and TP2, AR and MTP) all gave `25e35c11` for the first turn; not
  reproduced, not explained.
- **Not measured:** eviction on hardware (the budget follows host memory and
  has no switch; the hosted test covers it), and Q8 with reuse.

## Plan

1. **Executor**: done, with sampled MTP and cache reuse (above).
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
  there before 2026-09-26 afternoon checked old code. Copy the tree into the
  container (`git ls-files -z | tar --null -T - -cf - | podman exec -i -w
  /tmp/fmt nixbox tar -xf -`, after `git init` there) and run `nix
  --extra-experimental-features "nix-command flakes" shell --inputs-from .
  nixpkgs#clang-tools nixpkgs#python3 -c python3 tools/ci/check-format.py`.
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

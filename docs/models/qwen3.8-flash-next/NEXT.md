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
| `claude/tp2-cache-reuse` | none | working branch: everything above plus the work below |

The working branch carries production changes that belong in #2 and are not in
it: the faster exchange (GPU-side add, deferred ack), the idle control-channel
keepalive, the Q8 cross-rank fix, the shared-expert split, MTP with up to seven
drafts, and rank 1 as an executor with sampled MTP and cache reuse. It also
merges the Q8 PLE loader from #1 and carries the C2 probe, measurements and
documentation.

## What works

Two-host serving (`fuzzy` rank 0, `misty` rank 1, FDR InfiniBand) of one
request at a time, Q4 and full Q8, AR and MTP, with sampling, streaming, stop
sequences, client cancellation and history reuse as on one host. Both hosts
stay bit-identical; lost peers, scope mismatches and rank disagreement fail the
request with HTTP 500. Full Q8 needs both hosts; it does not fit one.

| Decode, tok/s | Q4 TP2 | Q4 one host | Q8 TP2 |
|---|---|---|---|
| AR, one stream | 26.9 | 25.9 | 24.2 |
| MTP greedy, mixed / repetitive | 41.7 / 43.0 | 38.7 / 46.4 | 37.4 / 42.3 |
| MTP sampled (T 0.7, top-p 0.95), mixed / repetitive | 35.5 / 38.5 | 33.1 / 34.4 | |
| AR batched, width 2 / 4 / 8 (probe only) | 44.8 / 63.2 / 87.7 | 45.7 / 76.3 / 108.7 | 37.2 / 54.4 / 76.2 |

Prefill from 0 to 32K runs at 1,187 tok/s on TP2 against 1,422–1,457 on one
host. A follow-up turn on 5.9K tokens of history reaches its first token in
124–141 ms instead of about 4.9 s. Single-host batched figures are HTTP sums at
2K prompts, so that row is a reference rather than a matched control.

Why TP2 is not faster than one host on Q4 (profile of both ranks):

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

- **Concurrency.** One request at a time; see [C2](#next-c2-on-the-executor).
- **Q8 quality.** The ranks agree with each other, but full Q8 has not been
  compared with a reference. It needs a CPU or reference-logit comparison,
  since Q8 does not fit one host.
- **Cache reuse, open points.** Eviction is covered only by the hosted test
  (the budget follows host memory and has no switch), and Q8 has not run with
  reuse. Once, before the cache fixes, one request gave two different stable
  outputs across server restarts; seven restarts since gave one output. Not
  explained.

## Next: C2 on the executor

Not started. The executor retired the dormant C2 worker path, which needed rank
1's own scheduler. What it takes now:

1. **Concurrent requests on rank 1.** `TpExecutor::RunRequest` serves one
   request from `kSingle` to `kEnd`, and rank 0 serializes requests with
   `tp_request_mutex`. Rank 1 needs one loop that keeps every open request
   (prompt, sampler replica, digest, count) and dispatches instructions by
   sequence; `kEnd` then closes one request while others continue.
2. **Operation scope per call, not per request.** The communicator allows one
   active scope, bound to a request's sequence. Bind it per instruction
   instead (the channel-wide index is already known to both ranks).
3. **Batched instructions.** Mirror `AdvanceBatch` as one instruction listing
   `(state, token)` in order, and remove the Flash-Next runner's refusal of
   batched calls under TP2. The batched AR forward with exchanges already ran
   in the probe up to width 8. Batched MTP (`DecodeBatch`) has never run under
   TP2: keep MTP requests on serial decode steps until it is qualified.
4. **Per-request failure.** A rank-1 failure now clears rank 0's whole cache,
   which is impossible while other requests hold leases. Invalidate only the
   failed request's state and snapshots instead.
5. **Limits.** Raise the response broker's capacity and lift the one-session,
   one-pending, one-connection refusals together.

Expected value: the probe measured Q4 TP2 below one host from width 4 (63.2
against 76.3 tok/s), so C2 mainly serves full Q8, which does not fit one host,
until the speed items above land.

## Plan

1. **Q8 quality** against a reference, and the `slow`/`external-model` suites.
2. **Upstream.** Open an issue asking whether two-host InfiniBand TP is wanted:
   it adds a libibverbs dependency and hardware upstream likely cannot test,
   and the build gate must stay off by default. Then move the working-branch
   changes into #2, condense it into a short series, and submit TP2 C1 and the
   Q8 PLE loader as separate PRs with a shared qualification.
3. **Speed** (list above).
4. **C2** (above). Park #3 until C2 has a caller.

Transport: InfiniBand only until this plan is done. RoCEv2 keeps the one-sided
RDMA-read design and mainly needs the right GID index.
[OdinLink-Five](https://github.com/Geramy/OdinLink-Five) (Thunderbolt RDMA)
also sits under libibverbs, but its provider sends every work request as a
two-sided SEND and has no `ibv_query_gid`, so it needs a two-sided exchange
behind the `Communicator` interface.

## Verification debt

- The `slow` and `external-model` suites have never run on these branches.
- Four Python tests fail without `numpy`, which the ROCm 7.2.3 dev image lacks;
  they also fail on an unmodified checkout.
- `LocalExpert` has no production caller. Its test expectation was corrected to
  the documented `global - expert_begin` rule; revisit if it gets one.
- `inference_backend_gpu_test` skips without a model; a skip is not a pass.

## Not claimed

- No C2 in serving; batched decode has run only in the probe, greedy, no MTP,
  one run per configuration.
- TP2 and one-host logits differ by reduction order and are compared with a
  tolerance, not bit-identity.
- Cached and uncached outputs can differ on one host and on TP2: a cached turn
  prefills a short suffix with different kernel shapes.
- Full Q8 is not a supported serving target.
- TP2 serves one request at a time.

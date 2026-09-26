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
greedy, non-streaming request at a time, Q4 and full Q8, AR and MTP. Both hosts
stay bit-identical; lost peers, scope mismatches and rank disagreement fail the
request with HTTP 500. Full Q8 needs both hosts; it does not fit one.

| Decode, tok/s | Q4 TP2 | Q4 one host | Q8 TP2 |
|---|---|---|---|
| AR, one stream | 26.9 | 25.9 | 24.2 |
| MTP, mixed / repetitive | 42.0 / 43.4 | 38.9 / 46.5 | 37.7 / 42.4 |
| AR batched, width 2 / 4 / 8 (probe only) | 44.8 / 63.2 / 87.7 | 45.7 / 76.3 / 108.7 | 37.2 / 54.4 / 76.2 |

Prefill from 0 to 32K runs at 1,187 tok/s on TP2 against 1,422–1,457 on one
host. Single-host batched figures are HTTP sums at 2K prompts, so that row is a
reference rather than a matched control.

Why TP2 is not faster than one host on Q4:

- Only routed experts are split, and they are about 13% of decode kernel time;
  dense Q8 GEMVs are 57% and stay replicated.
- TP2 adds about 3 ms per token: 48 exchanges plus a stream synchronization
  before each staging copy.
- Prefill is capped by the NICs' PCIe Gen3 x4 links (3.27 GB/s one way).
- From width 4 up, about 10 ms of each step is waiting for the rank whose
  experts received more rows, although the payload is only 40–80 KiB.

## What does not work yet

- **Ordinary clients.** Sampled, streaming and `stop` requests are refused
  ([TP2.md](TP2.md#request-limits)). Upstream's pending
  `feat/model-sampling-defaults` (#277) makes requests without an explicit
  temperature sampled (1.0 / top_p 0.95 / top_k 20); once it lands, every TP2
  request that does not send `temperature: 0` gets a 400.
- **Q8 quality.** The ranks agree with each other, but full Q8 has not been
  compared with a reference. It needs a CPU or reference-logit comparison,
  since Q8 does not fit one host.
- **Concurrency.** One request at a time; C2 is dormant.

## Plan

1. **Request-end protocol.** Rank 0 already decides stops, cancellations and
   client disconnects, and rank 1 always waits for the next step, so rank 0
   sends an end marker in place of the next token. One mechanism covers stop
   sequences, cancellation, disconnects and streaming, and nothing depends on
   both ranks evaluating stop rules identically. Shipping the rules instead is
   not enough: with the guard removed on two hosts, a `stop` request was
   silently ignored. Design rules:
   - Every exit path on rank 0 emits a terminal step: EOS, length, stop,
     cancellation and failure.
   - `final_token_advance_required` is false, so the last published token does
     not cause another forward; the end marker must not either.
   - Cancel only at an agreed step boundary, never by abandoning one rank's
     collective; cancellation during prefill needs a boundary of its own.
   - Chat Completions and Responses share one coordinator.
2. **Sampled AR.** Lift the sampling refusal for AR: rank 0 samples, rank 1
   consumes. Rank 1's own-choice check applies to greedy only.
3. **Step plan for MTP.** `Session::PrepareDecode` chooses the anchor, draft
   length and proposal chain, and `Session::FinishDecode` selects the accepted
   prefix, correction and rollback. These decisions shape the collectives
   before final tokens exist, so rank 0 must send them; its final tokens alone
   are not enough. This is also what sampled MTP and C2 with MTP need.
4. **Q8 quality** against a reference, and the `slow`/`external-model` suites.
5. **Upstream.** Open an issue asking whether two-host InfiniBand TP is wanted:
   it adds a libibverbs dependency and hardware upstream likely cannot test,
   and the build gate must stay off by default. Then move the working-branch
   changes into #2, condense it into a short series, and submit TP2 C1 and the
   Q8 PLE loader as separate PRs with a shared qualification.
6. **Speed.** Split the dense projections (the only change that can make TP2
   decode clearly faster than one host; it adds collectives), overlap prefill
   exchanges with compute, and reduce the rank wait at higher width by
   overlapping or by balancing expert placement from measured routing.
7. **C2 and concurrency**, once the above works. Park #3 until C2 has a caller.

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
- Neither TP2 target is usable with ordinary clients yet.

# Qwen3.8-Flash-Next TP=2

Two-host expert-parallel execution over InfiniBand. It is experimental and
build-gated (`GUFO_ENABLE_TP2_RDMA`), and it is not a published benchmark
configuration. This file is the reference: how TP2 works, how to run it and how
to qualify a change. Current status and the plan are in [NEXT.md](NEXT.md);
measurements are in [EXPERIMENTS.md](EXPERIMENTS.md).

## Design

**Partition.** Routed experts are split by rank (full Q8: rank 0 holds experts
0–255, rank 1 experts 256–511). The shared expert's intermediate dimension is
split in half. Dense, attention, GDN, HC, embedding, PLE and LM-head weights are
replicated. Both hosts hold the complete checkpoint; RDMA carries activations,
not weights. A rank with no locally selected experts contributes zeros and still
takes part in the exchange.

**Exchange.** Each MoE layer ends in one exchange of the ranks' partial outputs:
48 per token, 10 KiB per decode row. Each rank stages its partial in host
memory, reads the peer's partial with a one-sided RDMA read from a fixed IOVA
window, and adds it on the GPU. A small ordered TCP header precedes each
exchange and the read acknowledgement is deferred to the next one. Completion
uses an RDMA completion channel, with bounded CQ polling as a provider
fallback. HIP graph capture is disabled under TP2. Startup fails safely if a
fixed staging address is already occupied.

**Numerics.** The sum has two operands, so both ranks compute bit-identical
results and replicated work stays identical on both. Splitting experts changes
the reduction order relative to one host, so TP1 and TP2 logits are compared
with an explicit tolerance, never claimed bit-identical.

**Operation scope.** Every request binds an operation lease scoped to its
control sequence. Each exchange header carries the scope, a monotonic
per-collective ordinal and the byte count. Any mismatch poisons both
communicators and fails the request; a poisoned rank fails every later request
until it is restarted.

**Control channel.** An ordered, token-authenticated TCP channel (protocol v6)
carries prepared prompts, per-token steps and responses. The handshake
validates context, MTP use and draft width, cache policy and prefill chunk
size. Rank 0 owns a single response reader that routes frames by command
sequence; an unknown or duplicate sequence poisons the channel. After the
handshake an idle rank waits without a timeout, and TCP keepalive reports a
dead peer host.

**Rank 1 as executor.** Rank 0 runs the ordinary scheduler and runner pool
through `TpMirroredRunner`. Just before each call that changes model state, the
wrapper sends it to rank 1 as a `kInstruction` (state reset, prefill chunk,
advance, or one multi-token decode cycle); rank 1's `TpExecutor` makes
the same call on the same state, and both ranks meet in the call's exchanges.
Token selection reads only logits and stays on rank 0, so sampling, stop
sequences, cancellation and streaming are decided by rank 0's scheduler exactly
as on one host; rank 1 simply receives no further calls. A request starts with
`kSingle` and ends with `kEnd`. Rank 1 builds no pool or scheduler: it creates
as many states as rank 0's pool, so a state id names the same state on both
ranks. Rules that the compiler does not enforce:

- Every call that changes model state must pass through the wrapper, including
  state resets, which the continuation cache makes directly on the state. The
  base `TextModelRunner::DecodeStep` calls `SelectNext` and `Advance` on the
  runner it belongs to, so the wrapper implements the token-by-token path
  itself rather than forwarding it.
- The wrapper never passes a request's cancellation check to the model session,
  which can abort between layers and would strand rank 1 inside an exchange.
  The scheduler cancels between calls, so a long prefill stops at its next
  chunk.
- A multi-token decode cycle is one instruction. It carries rank 0's sampler
  draw state (RNG and pending deferred draw) from just before the call, and
  rank 1's sampler is built like the pool's (the request's sampling
  configuration, the prompt as history) and accepts the same tokens, so both
  ranks draw alike and make the same draft and acceptance decisions. The
  draft-length controller depends only on acceptance history
  (`mtp_policy.hpp`). A one-token budget selects on rank 0 and mirrors the
  advance.
- Cache operations are mirrored too, and acknowledged: see
  [Cache reuse](#cache-reuse).

**Agreement checks.** Both ranks keep a digest of every call and its result
(prefill consumption, decoded tokens and draft counts, checkpoint positions,
failures). `kEnd` carries rank 0's digest and call count, and rank 1 fails the
request when either differs. Under greedy decoding rank 1 also compares its own
argmax with every token rank 0 advances, computed after each call while rank 0
is still selecting, which catches a numerical divergence such as the full-Q8
bug even though both ranks feed rank 0's token. A failure is reported in rank
1's response and the request returns HTTP 500; the pair stays usable.

**Failure behaviour.** A lost peer surfaces within about a second as a TCP reset
on the retained bootstrap socket; rank 0 returns 500 and later requests fail on
the poisoned communicator. A scope mismatch fails on the first exchange, before
any output.

## Build

The adapter uses the pinned `rdma-core`/`libibverbs` and is off in the default
build.

```sh
nix develop --inputs-from .#tp2-rdma -c cmake --preset gpu-tp2
nix develop --inputs-from .#tp2-rdma -c cmake --build --preset gpu-tp2 --target gufo

# Release binary with TP2
cmake --preset release -DGUFO_ENABLE_TP2_RDMA=ON
cmake --build --preset release --parallel 4
```

On `fuzzy`/`misty` the same presets build inside the `gufo-tp2-dev:7.2.3`
container with the repository mounted at `/workspace/gufo`.

## Run

Start rank 0 first. `RANK0_ADDRESS` is the address rank 1 uses to reach rank
0's TCP bootstrap; tensor data uses the IB queue pair. Both ranks need the same
model files, `--tp-control-token`, `--prefill-chunk`, draft settings and cache
policy.

```sh
# rank 0: public HTTP server
build/gpu-tp2/gufo serve llm \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --speculative mtp \
  --mtp-model models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --tp-world-size 2 --tp-rank 0 \
  --tp-bootstrap-port 18515 --tp-control-port 18516 \
  --tp-control-token SHARED_TOKEN \
  --max-pending 1 --max-pending-per-client 1 --max-connections 1

# rank 1: worker only, no public HTTP port
build/gpu-tp2/gufo serve llm \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --speculative mtp \
  --mtp-model models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --tp-world-size 2 --tp-rank 1 --tp-bootstrap-host RANK0_ADDRESS \
  --tp-bootstrap-port 18515 --tp-control-port 18516 \
  --tp-control-token SHARED_TOKEN \
  --max-pending 1 --max-pending-per-client 1 --max-connections 1
```

For full Q8 use the first `Q8_0` shard; the Q8 PLE loader must be in the build.
Add `--tp-cache-reuse` on both ranks to reuse conversation history as one host
does; see [Cache reuse](#cache-reuse).

### Request limits

Sampling, streaming, stop sequences and client cancellation work as on one
host. The server still refuses:

| Refused | Why |
| --- | --- |
| More than one session, pending request or connection; request timeouts | Rank 1 executes one request's calls at a time. |
| Disk cache | Snapshots are rank-local and never serialized. |
| Vision input | Not implemented for two ranks. |

### Cache reuse

With `--tp-cache-reuse` rank 0's continuation cache works exactly as on one
host (live frontiers, prompt snapshots, restores, eviction), and every cache
operation that touches model state is mirrored:

| Rank-0 call | Instruction | Rank 1 |
| --- | --- | --- |
| `Snapshot(state)` (pool capture worker) | `snapshot(state, id)` | captures its own copy under `id` |
| a mirrored snapshot is freed | `drop(id)`, also between requests | frees its copy |
| `RestoreOrFork(state, snapshot)` | `restore(state, id)` | restores its copy |
| `PreparePrefixReuse(state, prefix)` | `reuse(state, prefix_size)` | the same call on the `kSingle` prompt |
| `PrepareCancellation(state)` | `cancel-prepare(state)` | the same call |

Live hits need nothing more: every call that changed the state was mirrored.
Snapshot bytes stay on each host; ids are channel-wide and never reused. The
snapshot budget is the smaller of the two hosts' `HostSnapshotBudgetBytes`,
exchanged in the handshake.

Rules:

- Rank 1 acknowledges each of these operations (except `drop`) before rank 0
  returns from the call, so no later instruction can overtake a capture and a
  rank-1 failure is known before the next call. Rank 0 runs its own copy while
  rank 1 runs its.
- A capture that fails on either rank skips the snapshot, as a failed capture
  does on one host: rank 0 drops the id and the request continues. Any other
  cache failure on rank 1 fails the request, and rank 0 then clears its whole
  cache, which resets and drops everything on both ranks.
- The capture runs on the pool's worker thread while the pool keeps the state
  frozen, so no call on that state can be sent before the capture is joined.
- The digest includes snapshot ids and each restored or reused position.
- TP2 must not change the cache's boundaries: an earlier design, where each
  rank kept its own cache, cut the prompt snapshot before the generation
  prompt, and TP2 then restored less than one host did.

Disk persistence stays refused. Cached and uncached outputs can differ on one
host too (prefilling a short suffix runs different kernel shapes), so compare
TP2 with one host under the same cache hits, not cached with uncached.

### Probes

`tests/models/qwen38_flash_next/` builds these with the `gpu-tp2` preset:

- `qwen38_flash_next_tp_probe`: prompt, decode and logit comparison on both
  ranks (`--tp-world-size 2 --tp-rank N`, `--tp-operation-id N` for a
  deliberate identity mismatch).
- `qwen38_flash_next_tp_c2_probe`: the C2 cohort contract and the batched-decode
  spike. `--batched-w2`/`--serial-w2` run the same program on both ranks,
  batched or one member at a time; `--width N` (2–8) sets the member count;
  `--allreduce-bench N` times the exchange without a model.
- `qwen38_flash_next_ple_gather_probe --prompt-file F`: host-only hash of the
  PLE rows a prompt gathers; diff stdout between hosts.
- `tp_instruction_latency_test` (label `perf`): loopback cost of one
  instruction.
- `tp_executor_test`: the real scheduler and pool on rank 0 against an executor
  on rank 1 over a loopback channel, with a toy model; checks that both ranks
  make the same calls and that divergences are reported.

### Benchmark driver

`tools/bench/model-bench.py` launches both ranks through an experiment-only
`gufo.tp2` overlay in a copied `bench.json`. Results are experimental and must
not be rendered into the published tables.

```json
{
  "gufo": {
    "tp2": {
      "remote_host": "misty",
      "bootstrap_host": "RANK0_REACHABLE_ADDRESS",
      "bootstrap_port": 18515,
      "control_port": 18516,
      "control_token": "EXPERIMENT_ONLY_TOKEN",
      "cache_reuse": false,
      "container_image": "gufo-tp2-dev:7.2.3",
      "workspace": "/path/to/gufo",
      "container_workspace": "/workspace/gufo",
      "binary": "build/gpu-tp2/gufo"
    }
  }
}
```

```sh
python3 tools/bench/model-bench.py --model qwen3.8-flash-next \
  --gufo build/gpu-tp2/gufo --config /tmp/bench-tp2.json \
  --artifacts-dir /tmp/gufo-tp2-artifacts --gguf "$MODEL" --mtp "$MTP" \
  run --target gufo --table single-mtp --depths 0 --context 4096 --mode mtp
```

The driver uses non-streaming JSON requests, records both rank fingerprints,
redacts the token and bootstrap address, and refuses loading, memory, image,
C>1 and (unless `cache_reuse` is set) nonzero-depth tables. `multi-*` tables
run only at concurrency 1 with uncached requests.

## Full Q8 checkpoint

`/opt/models/qwen3.8-flash-next/Q8_0/` on both hosts: six shards, about 188 GB.
SHA-256, computed independently on both hosts:

| Artifact | SHA-256 |
|---|---|
| `Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf` | `2dabcbb53ca537a7947bc7d20414fd464eeaf4d66d43021b5b2556cc87544ad2` |
| `Qwen3.8-Flash-Next-Q8_0-00002-of-00006.gguf` | `494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e` |
| `Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf` | `34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20` |
| `Qwen3.8-Flash-Next-Q8_0-00004-of-00006.gguf` | `bfa634025fabbd2658bf7694bc80b90e571699c768723f844c934c7ef06c691a` |
| `Qwen3.8-Flash-Next-Q8_0-00005-of-00006.gguf` | `232a8f14cc0fa4262e7efe8593774b136fe40909e39c7a020342ddaa27259a97` |
| `Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf` | `538a93bca918064983409a41187ad4c68640f9aced6f29564da8f551bf86d7a5` |
| `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6` |

The PLE table `per_layer_token_embd.weight` (shard 3, `[320001536, 160]`, Q8_0,
170-byte rows, about 54.4 GB) stays host-side and is read through direct I/O on
each host; it needs the Q8 PLE loader. Full Q8 does not fit one 128 GiB host:
at startup each TP2 rank uses 71,449 MiB of GPU memory, 72,856 MiB with MTP.

## Qualifying a change

Run this on both hosts for any change to the TP2 path, and record the result as
one row in EXPERIMENTS.md (machine-readable detail in `artifacts/`).

1. Build the release binary and the test targets from one commit on both hosts;
   record the commit and binary hash.
2. Format check on a fresh `git archive` export of that commit, not on a mounted
   checkout that may be stale.
3. Hosted tests on each host, with no skips: `tp_control_test`,
   `tp_executor_test`, `tp_cohort_plan_test`, `tp_cohort_worker_test`,
   `text_model_runner_test`, `text_generation_scheduler_test`,
   `serve_cli_test`, `qwen38_flash_next.ngram`,
   `qwen38_flash_next.tp_partition`, `qwen38_flash_next.mtp_sampling`.
4. Serving on two hosts, greedy: Q4 and Q8, AR and MTP; a short prompt, its
   repeat, and a prompt longer than one prefill chunk. Outputs must repeat, and
   AR and MTP outputs must match within each quantization.
5. Request types: a seeded sampled request (repeats reproduce it), a streamed
   request, a `stop` request that ends at its stop sequence, and a client
   disconnect during decoding and during a long prefill. A valid request must
   succeed after each.
6. Failure paths: kill rank 1 mid-request (rank 0 returns 500 promptly); a
   scope mismatch fails on the first exchange; an injected rank-1 disagreement
   returns 500 and the next request succeeds.
7. With `--tp-cache-reuse`: a multi-turn greedy AR chat, its identical replay
   (restores the whole prompt) and a follow-up on about 6K tokens of history;
   outputs and cached token counts must match one host with its cache.
8. For speed, report the median of several warm requests; the first request is
   about 13% slower.

## C2 (dormant)

The control protocol still defines the two-member C2 cohort envelope, and the
cohort contract code and its tests remain, but rank 1 no longer dispatches C2
commands: it has no scheduler to admit a cohort into, and it stops if it
receives one. C2 returns on batched executor instructions (`AdvanceBatch`,
`DecodeBatch`). The C2 probe's batched-decode spike (`--batched-w2`,
`--serial-w2`, `--width`) does not use the worker and still runs; its cohort
contract modes relied on the retired dispatch. The open design decisions are
in [NEXT.md](NEXT.md).

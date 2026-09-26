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

**Rank-0 step plan (AR).** Rank 0 samples each token and publishes it as a
`kStep` command; rank 1 consumes it in `SelectNext` instead of sampling, so both
ranks run the stop test on the same token. Each consume is bounded by 5 s; a
clean timeout does not poison the channel. `TpStepArm` arms the channel in all
three request entry points (`GenerateScheduled`, `start_chat`, the rank-1
worker) and disarms it on every exit. Two ordering rules are not enforced by the
compiler:

- Rank 0's publisher must be armed before its scheduler thread can decode.
- The prefill preview (`PreviewFirstToken`, from `PrepareFirstSnapshot`) samples
  a token that prefill discards. It must not publish or consume on either rank,
  so `SelectNextImpl` takes `use_step_channel = false` for it.

The MTP path (`DecodeStep`) does not use the step channel: each rank drafts and
verifies on its own. The draft-length controller depends only on acceptance
history, never on timing (`mtp_policy.hpp`), so identical tokens give identical
draft lengths and identical collectives.

**Agreement checks.** When a request ends, rank 0 compares rank 1's tokens and
draft/cache telemetry with its own; any difference fails the request, since it
means the exchanges combined partials from different states. Under the step
plan rank 1 also compares its own greedy choice, made on a copy of its sampler,
with each consumed token; the first disagreement fails the request after both
ranks have finished its collective schedule. Both failures return HTTP 500 and
leave the communicator usable.

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
`--tp-cache-reuse` on both ranks enables symmetric rank-local live-prefix reuse
and one retained snapshot boundary; no snapshot bytes cross hosts.

### Request limits

The server refuses these with HTTP 400 (or at startup), because the TP2 path
cannot yet keep both ranks in step for them:

| Refused | Why |
| --- | --- |
| Sampling (`temperature` > 0) | Not enabled yet. The step plan makes sampled AR possible; sampled MTP also needs rank 0's draft and acceptance decisions. |
| Streaming | Rank 1 must learn that a request ended early; there is no end marker yet. |
| `stop` sequences | Rank 1 never receives the rules and would decode past rank 0's stop. |
| More than one session, pending request or connection; request timeouts | Both ranks must run one identical collective schedule. |
| Disk cache, vision | Not implemented for two ranks. |

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
- `tp_step_latency_test` (label `perf`): loopback cost of one step message.

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
   `tp_cohort_plan_test`, `tp_cohort_worker_test`, `text_model_runner_test`,
   `text_generation_scheduler_test`, `serve_cli_test`,
   `qwen38_flash_next.ngram`, `qwen38_flash_next.tp_partition`,
   `qwen38_flash_next.mtp_sampling`.
4. Serving on two hosts, greedy: Q4 and Q8, AR and MTP; a short prompt, its
   repeat, and a prompt longer than one prefill chunk. Outputs must repeat, and
   AR and MTP outputs must match within each quantization.
5. Refusals: a sampled, a streaming and a `stop` request each return 400, and a
   valid request succeeds afterwards.
6. Failure paths: kill rank 1 mid-request (rank 0 returns 500 promptly); a
   scope mismatch fails on the first exchange.
7. For speed, report the median of several warm requests; the first request is
   about 13% slower.

## C2 (dormant)

C2 runs two requests as one cohort under a single operation lease. What exists:
a validated two-member AR-only command envelope with execution/cache plan
digests (protocol v6), atomic two-member scheduler admission, a translation seam
to ordered member responses, and rank-1 dispatch that refuses a cohort when an
MTP sidecar is loaded or its runner capacity is not one. No producer builds a
C2 command, and TP2 loading requires one session, so the path is dormant. The
C2 probe runs the contract and the batched-decode spike on hardware. The design
decisions that remain are in [NEXT.md](NEXT.md).

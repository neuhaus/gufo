# TP=2 RDMA development probe

This is the first Qwen3.8-Flash-Next two-rank execution path. It is an
experimental qualification topology, not a published single-host benchmark
configuration.

## Build

The optional adapter uses the pinned `rdma-core`/`libibverbs` package and is
disabled in the default build.

```sh
nix develop --inputs-from .#tp2-rdma -c cmake --preset gpu-tp2
nix develop --inputs-from .#tp2-rdma -c cmake --build --preset gpu-tp2 \
  --target qwen38_flash_next_tp_probe qwen38_flash_next_gpu_probe
```

The target and MTP sidecar must be present on both hosts. The first run uses
the verified `UD-Q4_K_XL` target and shared-Q8_0 MTP sidecar.

## Two-host probe

Run rank 0 first. `BOOTSTRAP_HOST` is the address rank 1 uses to reach rank
0's TCP metadata endpoint; model data itself uses the native IB QP.

```sh
# host 0
build/gpu-tp2/tests/models/qwen38_flash_next/qwen38_flash_next_tp_probe \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mtp-model models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --prompt 'The capital of France is' --tokens 16 --context 4096 \
  --tp-world-size 2 --tp-rank 0 --tp-bootstrap-port 18515

# host 1
build/gpu-tp2/tests/models/qwen38_flash_next/qwen38_flash_next_tp_probe \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mtp-model models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --prompt 'The capital of France is' --tokens 16 --context 4096 \
  --tp-world-size 2 --tp-rank 1 --tp-bootstrap-host RANK0_ADDRESS \
  --tp-bootstrap-port 18515
```

The loader discovers the remaining target shards beside the first shard. The
lower-level `qwen38_flash_next_gpu_probe` remains available for prefill/logit-dump
comparisons. Both TP probes accept `--tp-operation-id N`; pass the same value on
both ranks for a normal run. A deliberate rank mismatch is a negative identity
probe and must fail before producing model output.

## C1 serving qualification

After the probe passes, rank 0 can run the fail-stop HTTP worker path and
rank 1 can run the worker-only path. The first serving slice is intentionally
limited to one greedy, non-streaming, uncached request at a time:

```sh
# rank 0: public HTTP server
build/gpu-tp2/gufo serve llm \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --speculative mtp \
  --mtp-model models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --draft-tokens 1 --min-draft-tokens 1 \
  --tp-world-size 2 --tp-rank 0 \
  --tp-bootstrap-port 18515 --tp-control-port 18516 \
  --tp-control-token SHARED_TOKEN \
  --max-pending 1 --max-pending-per-client 1 --max-connections 1

# rank 1: worker only; this process does not bind the public HTTP port
build/gpu-tp2/gufo serve llm \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --speculative mtp \
  --mtp-model models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --draft-tokens 1 --min-draft-tokens 1 \
  --tp-world-size 2 --tp-rank 1 \
  --tp-bootstrap-host RANK0_ADDRESS \
  --tp-bootstrap-port 18515 --tp-control-port 18516 \
  --tp-control-token SHARED_TOKEN \
  --max-pending 1 --max-pending-per-client 1 --max-connections 1
```

The worker uses a separate ordered TCP control channel for prepared prompt
submissions. Both ranks must receive the same `--tp-control-token`, prefill
chunk setting (`--prefill-chunk`), and cache policy. RDMA remains the tensor
transport. Do not use this C1 path for streaming, cancellation, sampled
decoding, vision, disk continuation, or multiple concurrent requests yet. For
the controlled cache experiment, append `--tp-cache-reuse` to both rank
commands; the flag is explicit and is not enabled by the default TP2
configuration.

## Experimental benchmark driver

The official model-bench driver can launch the paired ROCm containers through
an experiment-only `gufo.tp2` overlay in a copied `bench.json`; the published
`artifacts/bench.json`, `BENCHMARKS.md`, renderer, and target variants remain
unchanged. The overlay requires `remote_host`, `bootstrap_host`,
`container_image`, `workspace`, `container_workspace`, `binary`, and a
non-empty `control_token`:

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

Set `cache_reuse` to `true` only for the controlled live-prefix/snapshot
experiment. It enables symmetric rank-local live-prefix reuse plus one
retained immutable boundary through the v4 control handshake (the prior
qualification was v3). Snapshot bytes
stay in each rank's host memory; no snapshot payload is sent over TCP or
RDMA. The default `false` setting preserves the uncached C1 boundary.

Run only the supported experimental d0 cells, with a separate artifact
directory:

```sh
python3 tools/bench/model-bench.py --model qwen3.8-flash-next \
  --gufo build/gpu-tp2/gufo --config /tmp/bench-tp2.json \
  --artifacts-dir /tmp/gufo-tp2-artifacts --gguf "$MODEL" --mtp "$MTP" \
  run --target gufo --table single-ar --depths 0 --context 4096 --mode ar

python3 tools/bench/model-bench.py --model qwen3.8-flash-next \
  --gufo build/gpu-tp2/gufo --config /tmp/bench-tp2.json \
  --artifacts-dir /tmp/gufo-tp2-artifacts --gguf "$MODEL" --mtp "$MTP" \
  run --target gufo --table single-mtp --depths 0 --context 4096 --mode mtp
```

For a live-prefix depth experiment, use the cache-enabled overlay and select a
single-user depth explicitly:

```sh
python3 tools/bench/model-bench.py --model qwen3.8-flash-next \
  --gufo build/gpu-tp2/gufo --config /tmp/bench-tp2-cache-reuse.json \
  --artifacts-dir /tmp/gufo-tp2-cache-depth --gguf "$MODEL" --mtp "$MTP" \
  run --target gufo --table single-ar --depths 4096 --context 8192 --mode ar
```

The qualified d4096 probe reused 4095 cached prompt tokens and prefilled 2043
new tokens. A paired historical-branch probe also restored the older stable
prompt boundary after a live continuation; both ranks reported the same
cached-token count, and MTP retained its draft acceptance. Clean-source MTP
d4096 measured 31.68 tok/s mixed and 39.07 tok/s repetitive in the fresh
post-v3 run; these remain experimental results, not published cells.

For the C1 corpus path, use an experiment configuration whose selected
`multi-ar`/`multi-mtp` concurrency is `[1]`; the driver forces
`prefill_first=false` and `cache_prompt=false`:

```sh
python3 tools/bench/model-bench.py --model qwen3.8-flash-next \
  --gufo build/gpu-tp2/gufo --config /tmp/bench-tp2-c1-multi.json \
  --artifacts-dir /tmp/gufo-tp2-c1-multi --gguf "$MODEL" --mtp "$MTP" \
  run --target gufo --table multi-ar --context 4096 --mode ar
```

The driver uses the JSON non-streaming request profile because the current TP2
server rejects streaming. It records a topology block, both rank fingerprints,
redacts the control token and bootstrap address, and rejects loading, memory,
image, C>1 concurrent, and nonzero-depth tables before launch. C1 `multi-ar`
and `multi-mtp` cells are supported only with forced uncached requests; the
current server cannot replay a distributed prepared prefix. These artifacts are
experimental and must not be rendered into the published benchmark tables.

## Current boundaries

- Expert-parallel routed MoE with replicated dense, attention, GDN, HC,
  embedding, and LM-head weights.
- Rank 0 owns the shared expert; the routed layer output is reduced with the
  ordered communicator.
- Host-staged synchronous RDMA reads use identical fixed IOVA windows for
  send, receive, and result data, with a small ordered TCP control/ack
  channel; an RDMA completion channel is used when available, with bounded
  CQ polling as a provider fallback. HIP graph capture is disabled.
  The probe fails safely if a fixed staging address is already occupied.
- The initial host sum changes reduction order, so distributed logits are
  compared with an explicit tolerance rather than claimed bit-identical.
- A rank with no locally selected experts emits a zero routed contribution
  and still participates in the collective.
- The serving runner marks distributed TP2 as serial-only. The CLI rejects
  `--sessions 2` before creating the communicator or loading weights. The
  default C1 path
  disables snapshots, forks, prefix reuse, persistence, and batched decode.
  `--tp-cache-reuse` enables symmetric rank-local live-prefix reuse and one
  immutable in-process snapshot/fork boundary; both ranks must use the flag
  and the control handshake rejects policy mismatches. No snapshot bytes cross
  the host boundary. The C1 path uses a separate rank-1 worker control channel
  and does not yet support streaming, cancellation, sampled decoding, or
  concurrent requests. Rank 0 now owns one control-response reader and routes
  final frames by command sequence; unknown or duplicate sequences poison the
  channel rather than being delivered to the wrong request. The C1 path also
  binds each RDMA collective to that command sequence and a per-collective
  operation ordinal; a scope, operation, size, or acknowledgement mismatch fails
  closed. This remains a response- and operation-ownership foundation only; it
  does not enable C2.
- Serialized/distributed disk snapshots, arbitrary historical-prefix indexes,
  and vision remain rejected. Only the stable prompt boundary and current live
  frontier are retained by this experimental C1 slice.
- C>1 is intentionally rejected before communicator/model setup. Merely
  changing `--sessions` is unsafe: a future cohort implementation must add a
  cohort identity and ordered member plan, an ordered rank-local execution
  plan, and explicit cache-plan parity before enabling physical batched
  collectives. A single member request sequence is not a valid shared C2
  collective scope.
  The first qualification slice, if pursued, is fixed two-request C2, AR before
  MTP, uncached and non-streaming, with repeated ordering/failure tests and
  per-request hashes; it must not be enabled by a CLI alias alone.
- C2 qualification is landing as three ordered layers, none of which enables a
  shared collective or a second session:
  - Protocol v5 validates a dormant two-member AR-only cohort envelope, canonical
    execution/cache plan digests, ordered member results, and broker mismatch
    poisoning.
  - The scheduler atomically admits exactly two ordered, unique members under one
    cohort identity, and rejects incomplete, duplicate, third-member, sampled,
    streaming, cached, continued, cancelled, and deadline-bearing cohorts before
    any model work. Admitted members still select `serial-c1` execution.
  - A translation seam converts a validated command into ordered scheduler
    members and two member results into a strict C2 response, rejecting any
    member that does not report serial width and zero draft/cache telemetry. A
    rejected cohort still returns both member envelopes.

  The rank-1 worker dispatches `kCohort2Ar` through those layers under exactly
  one operation lease scoped to the command sequence, and fails closed when an
  MTP sidecar is loaded because drafts cannot be represented by the C2 response
  contract. That path stays dormant: no producer constructs a C2 command. The
  sequence itself is host-testable through injected lease, submission and send
  hooks, with a single-release-site guard so a bound scope is released exactly
  once on every path — including a member failure, which would otherwise leave
  a scope bound and fail-stop the next C1 request.

  Not yet exercised on hardware: no C2 command has run end to end, because a
  driver would have to act as a full rank-0 peer (RDMA bootstrap plus control
  handshake) and execute both members in lockstep with the rank-1 worker.
- MTP follows the same expert partition and collective path.
- Q8_0 uses the same partition/upload contract; `gpu_probe` reports exact
  routed source bytes and the device model reports post-conversion resident
  bytes. The larger target must pass that local-memory fit check before it is
  promoted to a serving target.

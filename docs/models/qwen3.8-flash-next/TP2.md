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
  --tp-world-size 2 --tp-rank 0 \
  --tp-bootstrap-port 18515 --tp-control-port 18516 \
  --tp-control-token SHARED_TOKEN \
  --max-pending 1 --max-pending-per-client 1 --max-connections 1

# rank 1: worker only; this process does not bind the public HTTP port
build/gpu-tp2/gufo serve llm \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --speculative mtp \
  --mtp-model models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
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
  contract. It also refuses a cohort, before binding the lease, unless its
  runner capacity is one. A wider pool would admit both members at once and
  interleave them, and each rank's own scheduler decides that interleaving, so
  the two ranks' collective sequences could diverge. TP2 loading already
  requires one session, so this guard only matters once `--sessions 2` is
  enabled. That path stays dormant: no producer constructs a C2 command. The
  sequence itself is host-testable through injected lease, submission and send
  hooks, with a single-release-site guard so a bound scope is released exactly
  once on every path — including a member failure, which would otherwise leave
  a scope bound and fail-stop the next C1 request.

  Three real fault injections are now verified on hardware, all fail-closed:
  - **Admission refusal** (rank 1 `--worker-max-pending 1`): `SubmitCohort`
    throws at admission, the seam releases the bound lease and sends a
    wire-valid C2 error response that still carries both member envelopes.
    Rank 0 observed `members=2`, error `text generation pending queue is full`
    and no collective at all — 2 ms from command sent to response.
  - **Operation-scope mismatch** (rank 0 `--tp-scope-override N`): the first
    `AllReduceSum` header exchange fails on `scope_id` (`verbs.cpp:644-660`),
    poisoning both communicators. Rank 0 failed 13 ms after send with
    `outgoing=99/0/51200 incoming=1/0/51200`; rank 1, which injected nothing,
    detected the mismatch and then failed its own lease release with
    `verbs communicator is poisoned`, so the worker stopped instead of
    degrading quietly. This is the case a fake lease cannot model, and it
    confirms the exactly-one-`End` contract against a real poisoned
    communicator.

  - **MTP refusal** (rank 1 `--mtp-model`): TP2 plus MTP is a supported
    shipping combination, and the C2 response contract cannot carry draft
    telemetry at all, so a cohort must be refused rather than answered
    malformed. With a real 2.79 GB sidecar loaded, the handshake presented
    `use_mtp=true`/`max_draft_tokens=7`, rank 0 mirrored both fields, and rank 1
    refused **before** `BeginOperation` — no lease bound, no collective. Rank 0
    observed both member envelopes and the exact refusal string in 1 ms. The
    probe compares that string verbatim, because every refusal path returns the
    same exit code and a mode-1 refusal would otherwise read as this mode's
    success.

  Still covered by hosted tests only: peer cancel mid-cohort, member failure,
  and the runner-capacity refusal. TP2 loading rejects more than one session,
  so that refusal cannot be injected on hardware without relaxing the guard.
  No C2 command has run with a prompt longer than one prefill chunk.

- The dormant C2 path HAS now run end to end on hardware, three consecutive
  times, via `qwen38_flash_next_tp_c2_probe` (rank 0 on `fuzzy`, rank 1 on
  `misty`). `gufo serve` cannot host the worker: TP=2 forces `--max-pending 1`
  while `SubmitCohort` needs `queued + 2 <= max_pending_requests`, so a cohort
  is always refused at admission. The probe therefore hosts a real
  `InferenceBackend` with `max_pending_requests = 2`, runner pool capacity 1 and
  no MTP, and relaxes nothing in production. Per run: both digests matched, the
  cohort was admitted, member 0 then member 1 executed serially in one operation
  scope, and the ordered two-member response agreed token-for-token with rank 0's
  local greedy output. Each member issued 8 forwards and 384 collectives
  (48 layers), where 8 published tokens cost 7 decode advances because
  `final_token_advance_required` is false on the distributed runner. Still
  unproven: batched/physical C2, any performance benefit, and failure injection
  on hardware.
- MTP follows the same expert partition and collective path.
- Q8_0 uses the same partition/upload contract; `gpu_probe` reports exact
  routed source bytes and the device model reports post-conversion resident
  bytes. The larger target must pass that local-memory fit check before it is
  promoted to a serving target.
- Batched single-token advance was compared against serial advance on ONE host,
  Q4 UD-Q4_K_XL, greedy, 64 new tokens, two prompts of 2106 and 78 prompt
  tokens. `gufo serve -j 2` selected `plan=batched-w2 batch_width=2` with both
  requests co-resident (`resident_at_admission=2`, 427 ms queue wait), and the
  `-j 1` control logged `plan=serial-c1 batch_width=1`. All three paths --
  `gufo prompt`, `serve -j 1`, `serve -j 2` -- produced identical
  `reasoning_content`: sha256 `e83f2652...` (282 chars) and `731fb60a...`
  (353 chars). Batching the advance therefore does not perturb greedy tokens
  here. Scope, stated narrowly because it is narrow: the advance is the ONLY
  operation batched on this path. `Prefill` is per-state
  (`text_model_runner.hpp:262`) and has no batch form in the runner or the
  engine, so prefill ran serially in every arm and this result says nothing
  about prefill. `DecodeBatch` fell back to the serial base loop because
  `use_mtp_` was false (`inference_backend.cpp:2802`), so batched decode is
  untested. Single host, so no AllReduce and no distributed reduction order; 64
  greedy tokens, all of them `reasoning_content`; one run per prompt. Reference
  numbers only, not a like-for-like comparison: `prefill_tps=950.1` at width 2,
  and `cache_snapshot_bytes` 172425656 and 121428488 per resident session,
  which bounds how wide a batch can ever be.
- The same batched advance now also runs across two hosts. With
  `qwen38_flash_next_tp_c2_probe --batched-w2`, both ranks produce identical
  tokens and per-step member sets, including the routed-expert all-reduce inside
  `MoeBatch`, and match the `--serial-w2` baseline token for token. Batching two
  streams gives 1.67× the serial aggregate throughput. For Q4 the TP2 decode and
  prefill rates are no better than one host; see the performance section of
  [NEXT.md](NEXT.md#measured-tp2-performance) and `EXPERIMENTS.md`.
- **Rank-0 step plan, per-token exchange cost.** The step plan replaces
  independent per-rank sampling with rank 0 sampling and rank 1 consuming, so
  every token costs one small control message and the ranks run in lockstep.
  `tp_step_latency_test` measures that exchange over a loopback TCP pair on one
  host: **1,517,045 steps/s** pipelined, publish and consume ~0.6 µs each, and a
  **serialized round trip of 3.8 µs p50 / 5.3 µs p99**, i.e. ~1.9 µs one-way.
  Against a 26 tok/s Q4 decode (~38 ms per token) that is ~0.005%, and even a
  pessimistic 10× for the real cross-host path is ~0.05%, so lockstep is not a
  throughput tax. The `max` outliers (56-71 µs) are container scheduler
  preemption, not protocol cost.
  **This is a lower bound and not the deciding number.** It excludes the
  fuzzy/misty InfiniBand round trip and the peer host's scheduling delay; the
  cross-host figure has to come from a two-host run. The measurement asserts no
  threshold and is labelled `perf`, so it is excluded from the correctness run
  with `ctest -LE perf`; a slow result is a result, not a failure. Protocol,
  bounded receive and both role bridges are implemented, hosted-tested, and now
  verified on two hosts; see "Rank-0 step plan: verified on two hosts" below.

### Rank-0 step plan: verified on two hosts

The step plan is **implemented and verified on hardware**. Rank 0 samples each
token and publishes it as a `kStep` control command; rank 1 consumes that token
in `SelectNext` instead of sampling its own. Because the token originates on
rank 0, the stop test runs on the shared token and both ranks stop together with
no separate stop message, and the two ranks no longer need bit-identical logits.

Measured on `fuzzy`/`misty`, Q4 UD-Q4_K_XL, greedy, 64 tokens, AR, no MTP:

- rank 0 published 64 tokens, rank 1 consumed 64, first and last agreeing
  (`1596` ... `4971`), `sequence=1`, both ranks armed and cleanly disarmed
- **27.44 tok/s, `execution_plan: serial-c1`**, output byte-identical to the
  step-plan-off control at 27.3 tok/s

So the mechanism is proven rather than inferred, and the exchange costs nothing
measurable: the per-token command is ~1.9 us one-way (a loopback lower bound,
see the exchange-cost entry above), about 0.005% of a 26 tok/s token.

**Steady-state baseline.** Ten consecutive identical greedy requests on one
running pair, Q4 UD-Q4_K_XL, 64 tokens, AR, no MTP:

| | tok/s |
| --- | --- |
| first request (cold) | 23.17 |
| median of 10 | **26.69** |
| mean of 10 | 26.35 |
| max | 26.86 |

All ten produced byte-identical output, sha256 prefix `ea5625561f01b420`, and
zero divergence warnings on either rank. Nine of the ten fall in 26.63-26.86; the
first request is about 13% slower cold, so a single sample understates the
steady state and any later comparison should use the median of several runs
rather than one. This is the baseline to compare a change against: the
step-plan-off control measured 27.3 tok/s, so the exchange is within noise of
independent per-rank sampling at this width.

**What two-host runs cost to get here.** Four bugs, all in this work, found only
on hardware because each one is invisible to a hosted test:

1. **`SO_RCVTIMEO` leaked on the failure path.** `ReceiveCommandWithin` cleared
   the per-token bound only on success, so a timed-out consume left a 5 s timeout
   on the socket and rank 1's *idle command wait* — which must wait
   indefinitely — failed with `EAGAIN` and the worker exited.
2. **Clean timeouts poisoned the channel.** A receive that expired with nothing
   read leaves the byte stream intact, so poisoning it was both wrong and
   unnecessary; only a failure that may have landed mid-frame may poison.
3. **Rank 0 deadlocked on the response.** It waited for rank 1's reply before
   waiting on its own decode, so under the step plan each rank waited on the
   other. The order is now: send the command, submit, wait on rank 0's own
   decode, then collect rank 1's response.
4. **The step channel was armed in the wrong function.** TP2 has three request
   entry points -- `GenerateScheduled`, `start_chat` and the rank-one worker --
   and the arming existed in only one. HTTP goes through `start_chat`, so rank 0
   never armed its publisher at all and rank 1 starved. `TpStepArm` is now a
   single RAII type used by all three, so a new entry point cannot forget the
   disarm; `ScheduledGenerationRequest` holds it as a member because
   `start_chat` returns lazily and decodes inside `Wait`.

A fifth defect was a plain counter bug with an unhelpful symptom: the step
validator rejects `sequence == 0`, and `tp_sequence` started at 0, so the first
request a server ever served had every per-token message refused. The peer simply
starved. The counter now starts at 1, with a hosted test.

**How the failures were isolated.** A two-host bisect on one binary, with the
step channel behind an env gate, against the base branch as control:

| Arm | Reordering | Step channel | Result |
| --- | --- | --- | --- |
| base `d73562ab` | absent | absent | 27.3 tok/s, `serial-c1` |
| A `69cfcab1` | on | off | 27.3 tok/s, `serial-c1` |
| B `69cfcab1` | on | on | fail, 30.5 s, `bootstrap receive` |

Arm A matching the control exactly is what proved the reordering and the
own-wait ordering were correct and the step channel alone was at fault. Arm B then
failed on the *collective*, not the control channel, which is what pointed at
prefill and the preview rather than at the exchange. The env gate and the
`[step-trace]` tracing that made this possible are **removed**; they were
diagnostic scaffolding and must not ship in a serving path. Re-adding tracing is
the first step if the plan ever regresses.

One ordering hazard is worth keeping, because it is not enforced by the compiler:
the publisher must be armed before rank 0's scheduler thread can decode. Both
ranks run a speculative prefill lookahead (`PreviewFirstToken`, called from
`PrepareFirstSnapshot` at `text_generation_scheduler.cpp:818`) which delegates to
`SelectNext`. It samples a token the prefill path discards, so it must **not**
participate in the exchange on either rank; `SelectNextImpl` takes an explicit
`use_step_channel` flag for exactly this, and `PreviewFirstToken` passes false.

**Not yet done.** The request restrictions the plan unblocks are still enforced:
the `TP2 requires greedy text without stop sequences` guard still refuses
sampling, stop sequences and streaming, and the end-of-run token comparison is
still a gate rather than telemetry. That comparison is deliberate for now -- it
is the check that would catch a regression in the plan -- and demoting it is a
separate step, not something to fold in silently.

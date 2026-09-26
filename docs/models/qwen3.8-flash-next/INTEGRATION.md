# TP2 + Q8 baseline before rank-0 step execution

Verified on 2026-09-26. The implementation of the rank-0 step plan has **not**
started. This is the reproducible baseline and implementation handoff.

## Source and build

`codex/tp2-q8-integration` combines TP2 `d71fc6c` with Q8 PLE `426ab78` in merge
commit `0cf273a9124ebc63c08a123025075f3802bd25c9`. The merge was conflict-free.
The separate `pr/tp2-c1` and `pr/q8-ple` branches were not changed. They remain
separate upstream review units; their combination is the hardware qualification
target. The TP2 spike alone does not contain the Q8 PLE loader.

Both `fuzzy` (rank 0) and `misty` (rank 1) built that exact commit from
`~/git/gufo`, using the existing `gufo-tp2-dev:7.2.3` containers. The image IDs
differ, but the complete installed package manifests, compiler reports and
resulting binaries match byte for byte. GCC is 13.3.0, HIP is 7.2.53211 with
ROCm 7.2.3, and libibverbs is 50.0-2ubuntu0.2. This is the existing experimental
container toolchain, not a claim of qualification on the newer Nix production
toolchain described in the root README.

The production binary is `build/release/gufo`, SHA-256:

```text
fa52af6cfc78a3f3e138c1d00c85c3a083566296cccfc36d8b64bbd79740cf95
```

Build commands, inside the container with the repository at `/workspace/gufo`:

```sh
cmake --preset gpu-tp2
cmake --build --preset gpu-tp2 --parallel 4 --target gufo \
  tp_control_test tp_cohort_plan_test tp_cohort_worker_test \
  text_generation_scheduler_test text_model_runner_test \
  qwen38_flash_next_ngram_test qwen38_flash_next_tp_partition_test \
  qwen38_flash_next_mtp_sampling_test qwen38_flash_next_ple_gather_probe \
  qwen38_flash_next_tp_probe
cmake --preset release -DGUFO_ENABLE_TP2_RDMA=ON
cmake --build --preset release --parallel 4
```

## Verified behavior

- The pinned Nix format check passed for 490 C++ files on a fresh export of the
  merged index tree. Python was explicitly included in the Nix shell because
  the minimal `nixbox` container has none. The stale mounted checkout was not
  used for this check.
- Eight focused CTests passed on **each** host, with no skips: TP control,
  cohort plan, cohort worker, text runner, scheduler, Q8 n-gram reads, expert
  partition and MTP sampling. Documentation, dependency and Python serving
  benchmark harness checks also passed on both hosts.
- The real Q8 PLE gather matched on both hosts: 2,385 raw prompt tokens,
  38,160 gathered rows, three synchronous passes and an asynchronous cross-pass.
  All dequantized byte hashes matched. This is cross-host agreement, not an
  independent model-quality oracle.
- Four serving configurations passed: Q4 AR, Q4 MTP, Q8 AR and Q8 MTP. Each
  used the release binary, context 4,096, one session, greedy text, no streaming
  or cache, and a 64-token output budget. MTP had its normal seven-draft cap.
  Requests covered a 63-token prompt, its repeat, a 2,437-token prompt spanning
  two prefill work units, and a short request after invalid-request refusals.
  Each successful response generated 64 tokens. The server's existing checks
  verified rank token agreement and draft/cache telemetry. Short repeats also
  retained identical output and draft counts; AR and MTP outputs matched for
  both prompts within each quantization.
- All four configurations rejected sampled, streaming and stop-sequence
  requests with HTTP 400, then successfully served another valid request.
- During a Q8 MTP request with a 3,000-token budget, rank 1 was killed after
  two seconds while the request was still outstanding. Rank 0 returned HTTP
  500 within 0.354 seconds of initiating the kill. The error explicitly came
  through the response-broker drain path. A subsequent request returned HTTP
  500 immediately because the communicator was poisoned. This verifies the
  C1 broker failure path; it does not qualify graceful cancellation or C2.
- A separate Q4 RDMA probe used scope 41 on rank 0 and 42 on rank 1. Both
  exited unsuccessfully on the first collective, before generated output,
  reporting `41/0/51200` versus `42/0/51200`.

Startup GPU usage, equal on both ranks (not a peak-memory measurement):

| Target | AR | MTP |
| --- | ---: | ---: |
| Q4 | 47,061 MiB | 48,466 MiB |
| Q8 | 71,449 MiB | 72,856 MiB |

The test containers were removed. Pre-existing containers were left running.
These are bounded qualification runs, not benchmark cells. No full
`slow`/`external-model` suite, independent Q8 quality evaluation, deep-context
qualification, sampled TP2 or C2 serving is claimed. Model filenames and sizes
matched; the historical full-file checksums in `Q8.md` were not recomputed.

Retained machine-readable evidence is in
[artifacts/tp2-q8-integration.json](artifacts/tp2-q8-integration.json).
Full build logs, package manifests, request JSON, rank logs and the executable
test drivers are in the ignored repository directory
`artifacts/tp2-q8-integration-20260926/` on the editing host. A copy is retained
under the same directory's `qualification/` subdirectory on both test hosts.
The drivers use dedicated ports 18980/18915/18916 and remove only containers
they created. The control credential is temporary and is not retained.

## Next implementation slice: rank-0-controlled C1 AR

The first slice should keep width one and caching disabled. Rank 0 owns the
existing scheduler and sampler; rank 1 executes explicit model work and never
selects a token. Keep the existing greedy MTP path working until coordinated
MTP has its own tested execution program.

The relevant seams are:

| Responsibility | Existing implementation |
| --- | --- |
| Request admission, operation lease, final worker validation | `src/cli/serve/inference_backend.cpp`: `start_chat`, `GenerateScheduled`, `ScheduledGenerationRequest` |
| Worker currently starts its own greedy scheduler request | same file: `run_worker` |
| Existing model operations and token selection | same file: `QwenFlashNextTextRunner::{Prefill,SelectNext,Advance}` |
| Prefill, final-token, stop and cancellation decisions | `src/cli/serve/text_generation_scheduler.cpp`: `StepPrefill`, `PrepareDecode`, `CompleteIfStopped` |
| Framing, protocol v5, sole response reader | `src/cli/serve/tp_control.{hpp,cpp}` |
| Scope/ordinal/byte-count validation and RDMA exchange | `src/models/qwen38_flash_next/kernels/rocm/{communicator.hpp,verbs.cpp}` |

Implementation decisions to make concrete in the first patch:

1. Add a distinct C1 AR step-program command and an explicit protocol/capability
   agreement. Do not reinterpret dormant `kCohort2Ar` or silently downgrade to
   an independent worker scheduler. Validate the plan before model execution.
2. Represent begin, prefill range, advance token and finish decisions with a
   request sequence and monotonic step number. Include expected token position
   and validate every transition. Preserve room for an ordered member identity
   without enabling multiple sessions. A future batched step must also bind
   ordered membership to collective identity; equal byte counts alone do not
   detect swapped members.
3. Send each work instruction **before** rank 0 enters its corresponding
   forward pass. Rank 1 executes that operation through the existing model
   adapter. Wait for completion only after rank 0 has participated: waiting for
   the worker first would deadlock inside the paired collectives. Keep a single
   control-channel reader; route step receipts through it instead of adding
   competing socket reads in model callbacks.
4. Make rank 0's scheduler emit terminal decisions on every exit path: EOS,
   output length, stop sequence, cancellation and failure. In particular,
   `final_token_advance_required=false` means the final published token need
   not cause another forward. Broadcasting each sampled token unconditionally
   from `SelectNext` would therefore be wrong. Prefill cancellation must happen
   at an agreed work boundary, never by abandoning one rank's collective.
5. Finish and drain exactly once, retaining fail-closed behavior for lost peers,
   invalid steps and operation-scope errors. Hook both HTTP request entry paths
   into the same coordinator so Chat Completions and Responses do not diverge.

Start with host tests of the state machine and invalid ordering, then repeat
this hardware baseline with the new AR program. Next add seeded sampled AR,
streaming, stop sequences crossing token boundaries, disconnect cancellation,
and a successful request after each normal termination. Verify one-token
budgets, EOS and context exhaustion explicitly. Preserve the legacy MTP smoke
while the new C1 AR program is introduced.

MTP is a later slice: `Session::PrepareDecode` chooses the anchor, draft length
and proposal chain, and `Session::FinishDecode` selects the accepted prefix,
correction and rollback. Those decisions affect collective scheduling before
final tokens exist. Broadcasting only the final accepted tokens is insufficient.
Rank-0 token authority also does not replace numerical quality checks or prove
that independently computed MoE routing agrees. Keep the independent Q8 oracle
and routing-consistency questions explicit.

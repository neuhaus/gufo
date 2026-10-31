# Give pending TP snapshots a turn before AR advances

A pending snapshot can be starved by repeated autoregressive advances acquiring
the TP call mutex. On two Strix Halo nodes, an exact lock-site trace measured a
601.138 ms wait while 22 additional `Advance` calls acquired the same mutex. The
capture returned 3.875 ms after acquiring it. Other histories reproduced waits
of 2,239 and 888 ms. This is a host lock-handoff mechanism, separate from the
storage stalls investigated elsewhere.

The existing pending-snapshot gate covers multi-token decode. This follow-up
extends admission to `Advance`, `AdvanceBatch`, `DecodeStep`, and `DecodeBatch`.
An outer turn waits before taking the call mutex. A nested turn on the same
runner proceeds so an active decode can finish its recursive `Advance`; waiting
there would deadlock the capture needing that mutex. A thread-local linked
stack also handles nested turns across different runners and exception unwind.

An active call is allowed to finish, and prefill is unchanged. The check can race
with a newly registered capture, so this is not strict FIFO or a universal wait
bound. There are no model-math, GPU-kernel, collective-order, transport, cache
policy, wire-protocol, or system-configuration changes.

## Evidence and scope

The three-file follow-up is based on the existing snapshot-handoff branch
`b6d4cbaa`; its code commit is `92bf7cb6`. It is independently CPU-tested and
format-checked on that published base. The private production/inference test
stack also contains HC/cooperative/cache changes absent from this older base.
Hardware results below qualify that integration-stack delta, not a standalone
deployment of this published tree. Exact binary/source identities and settings
are in [qualification.json](qualification.json).

On the integration stack, the production build, sanitizer gate tests and five
CPU suites pass. An isolated negative control using unconditional nested waits
times out in the recursive-handoff test, demonstrating why nesting matters.
Actual two-node tests use Flash Next UD-Q4_K_XL, shared Q8_0 MTP when enabled,
65,536-token capacity, two sessions, CLI2K/backend4K and a 1 GiB cache budget.
Both hosts have 124 GiB HIP heaps and crossed 40+20 Gb/s links. The same temporary
driver is used in every paired arm and the approved drivers are restored.

### Untraced AR pilot

| Fixed execution order | Concurrent wave tok/s | Second-request first-followup gap ms |
|---|---:|---:|
| Parent before | 36.75 | 3342.25 |
| Fix first | 56.41 | 83.38 |
| Fix second | 56.52 | 83.87 |
| Parent after | 53.12 | 424.01 |

All 24 canonical request/output identities match. The first parent effectively
serializes the pair (physical width 1); the fix runs both requests together
(width 2). The first request therefore finishes later: 3662 ms versus 4512/4529
ms, while the second finishes sooner: 6956 ms versus 4528/4519 ms. Keep this
fairness/batching tradeoff distinct from a uniform latency improvement.
[All 22 timing flags](ar-phase-flags.csv) and
[four changed-width comparisons](ar-adaptive-work.csv) are retained. Controls
vary, and four fixed starts do not establish significance or stall frequency.
An instrumented parent/fix pair did not reproduce the large parent stall; it
does not prove a speedup.

### MTP and functional checks

The normal adaptive-MTP pilot has 24 matching canonical outputs, but concurrent
rates are 64.88/63.31 tok/s in controls and 57.27/68.99 in fix runs. Preserve
[25 timing flags](mtp-phase-flags.csv) and
[23 changed-work comparisons](mtp-adaptive-work.csv). There is no demonstrated
adaptive-MTP improvement or clearance of its slow candidate cohort.

A separate supported `--draft-tokens 1 --min-draft-tokens 1` diagnostic matches
all 12 outputs and recorded draft/acceptance/round/width counts, with no strict
timing flags: 71.00 versus 71.04 concurrent tok/s. This different workload does
not clear adaptive-mode flags or recommend a new default. The attempted floor
of two was rejected by Flash Next before model loading completed; it is not a
benchmark result.

Candidate functional coverage includes 23 HTTP requests: two concurrent
victim/survivor orders, exact survivor/post-cancel replay, and two deep seeded
continuations retaining/discarding interrupted assistant output. All six client
disconnects join by request ID to server `finish=cancelled` outcomes. Both deep
cases reuse 18,461 tokens and reproduce their seeded continuation and third-turn
outputs exactly. Actual two-row survivors and MTP proposals are checked.
Checked transport counters stay clean and tested servers exit. This does not
qualify process loss, cable loss, persistence or overnight operation.

## Review status

Keep this follow-up a draft. Remaining work includes adaptive-MTP timing
attribution, randomized concurrent-progress confidence, and hardware validation
of this standalone published base. Per-request flags remain open; no default
deployment, significance, universal deadlock clearance or blanket speedup is
claimed. Existing snapshot-handoff PRs retain their own review state.

## CPU reproduction

With the project's CPU test dependencies available:

```bash
cmake --preset cpu-test -B build-cpu
cmake --build build-cpu --target tp_control_test tp_executor_test tp_snapshot_gate_test text_model_runner_test text_generation_scheduler_test --parallel 4
ctest --test-dir build-cpu --output-on-failure -R '^(tp_control_test|tp_executor_test|tp_snapshot_gate_test|text_model_runner_test|text_generation_scheduler_test)$'
c++ -std=c++20 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined -pthread -I . tests/cli/tp_snapshot_gate_test.cpp -o /tmp/tp-snapshot-turn-test
timeout 20 /tmp/tp-snapshot-turn-test
```

Lab raw traces, per-request JSONL, exact frozen commands/settings, counter/kernel
windows and validators are preserved in `strix-dispatch-investigation` under
`hc-ar-snapshot-path-v1/v2`, `ar-snapshot-turn-trace-v1`,
`ar-snapshot-turn-pilot-v1`, `snapshot-turn-mtp-pilot-v1`,
`snapshot-turn-functional-v1`, and `snapshot-turn-mtp-fixed1-v1`.

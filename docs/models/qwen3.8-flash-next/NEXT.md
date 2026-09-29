# TP2 and full Q8: status and plan

As of 2026-09-29. Fork-only status for two-host TP2 and full Q8; update it when
the status changes and delete what it supersedes. How TP2 works and how to
qualify a change: [TP2.md](TP2.md); evidence: [EXPERIMENTS.md](EXPERIMENTS.md#tp2)
and [QUALITY.md](QUALITY.md#full-q8).

## Branches

| Branch | PR | Content |
|---|---|---|
| `feat/tp2-rdma` | [gufo-org/gufo#296](https://github.com/gufo-org/gufo/pull/296) | TP2 over InfiniBand RDMA, with batched MTP and the disk cache |
| `feat/qwen38-q8` | [gufo-org/gufo#297](https://github.com/gufo-org/gufo/pull/297) | full Q8 loader |
| `feat/tp2-roce` | none | RoCE v2, `--tp-rdma-device`/`--tp-rdma-port`; on top of #296; untested on RoCE hardware |
| `feat/qwen38-ple-lookahead` | none yet | next prefill chunk's n-gram rows read ahead; on upstream `main`, a candidate PR |
| `feat/tp2-systemd-units` | none | systemd user units, Containerfile |
| `rdma` | none | ready-to-use integration: upstream `main` + all of the above + TP2 benchmark tables; intended as the fork's default branch |
| `claude/tp2-next` | none | this file |

Explainer and benchmarks: [gufo-org/gufo#258](https://github.com/gufo-org/gufo/issues/258#issuecomment-5855543814).
Changes go to the feature branches first, then merge into `rdma`.

## Now

Local, not pushed: `feat/tp2-rdma` 02d97a9 and `rdma` 5ecf598 merge upstream
`main` with #282 (model sampling defaults); two harness conflicts resolved
(`tools/gufo/serving_bench.py`, `tools/gufo/model_bench/llm.py`). Both build on
the hosts; the harness test passes locally. When the hosts are free:

1. CPU suite and TP tests on both branches (`ctest -L cpu -LE perf`,
   `tp_control_test`, `tp_executor_test`); a sampled TP2 request with the new
   defaults (thinking off uses presence penalty 1.5); then push both.
2. Reproduce the published one-host prefill: Nix build of upstream `89d58eb`
   (the published pp's source), `model-bench run --target gufo --table
   single-ar` into a scratch `--artifacts-dir`; the published d0 pp is 1628.52.
   Our CMake `gpu-tp2` build measured 1351 on one host (decode equal): check
   whether the preset (gpu-test base, assertions) causes it; if so, switch
   the units and docs to a release build with TP2.
3. Nix build `.#tp2-rdma` of `rdma` (in `nixbox` on box1), copy its closure to
   `~/nixroot` on both hosts (mounted at `/nix` in the rank containers), then
   run `~/claude-runs/benchrun.sh` (one host with the same binary as
   `gufo-onehost`, TP2 Q4 as `gufo-tp2`, TP2 Q8 as `gufo-tp2-q8`); update the
   Nix store path in it and in `bench-tp2-q{4,8}.json` first. Render with
   `model-bench.py render` (matplotlib on box1) and commit the artifacts and
   charts to `rdma`.

## Open

1. **LM head split by vocabulary**: about 1.25 ms per token, with a small
   candidate exchange since rank 0 chooses the token.
2. **Prefill idle time**: after the n-gram lookahead, what remains per chunk
   is the last exchange (about 7 ms), the host turnaround between chunks
   (about 11 ms) and rank skew. HIP graphs (about 1%) and half-size exchanges
   (1–4%) were measured and rejected.
3. **Q8 batched MTP at C8** gains nothing over single-token batches (98.9
   against 99.7 tok/s); Q4 gains 9% there.
4. **Q8 against an official reference**: Gufo and llama.cpp differ by KL 0.09
   on Q4 and Q8 alike; which is closer to the original model is open.
5. **Sampled output differs after merging upstream `main`**: one seeded case
   (not from #292, which explains the greedy differences); not yet attributed.
6. **Poisoned pair**: two production crashes (2026-09-28) were not
   reproduced by a 10-minute cancel storm; the first poisoning error is now
   logged (`event=tp_worker_failed reason=…`).

## Not verified

- RoCE v2 on RoCE hardware.
- Cache eviction under TP2 only in the hosted test (the budget follows host
  memory and has no switch).
- The disk cache on hardware only through restart and restore; rank 1's
  eviction and a missing or corrupt rank-1 half only in the hosted tests.
- The other models' `external-model` tests need weights the two hosts lack
  (Qwen3.8 27B, DeepSeek V4 Flash); the shared serving changes are covered for
  them by the CPU scheduler, runner and cache tests.

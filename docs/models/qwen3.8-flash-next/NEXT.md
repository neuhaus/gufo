# TP2 and full Q8: status and plan

As of 2026-09-27. Fork-only status for two-host TP2 and full Q8; update it when
the status changes and delete what it supersedes. How TP2 works and how to
qualify a change: [TP2.md](TP2.md); evidence: [EXPERIMENTS.md](EXPERIMENTS.md#tp2)
and [QUALITY.md](QUALITY.md#full-q8).

## Branches

| Branch | PR | Content |
|---|---|---|
| `feat/tp2-rdma` | [gufo-org/gufo#296](https://github.com/gufo-org/gufo/pull/296) | TP2 over InfiniBand RDMA, with batched MTP and the disk cache |
| `feat/qwen38-q8` | [gufo-org/gufo#297](https://github.com/gufo-org/gufo/pull/297) | full Q8 loader |
| `claude/tp2-next` | none | all of them merged, plus this file; work continues here |

Explainer and benchmarks: [gufo-org/gufo#258](https://github.com/gufo-org/gufo/issues/258#issuecomment-5855543814).
Review changes go to the PR branches first, then merge into `claude/tp2-next`.
`feat/tp2-disk`, `feat/tp2-batched-mtp` and `feat/tp2-disk-cache` on the fork are superseded by #296.

## Open

In order of payoff for effort:

1. **LM head split by vocabulary**: about 1.25 ms per token, with a small
   candidate exchange since rank 0 chooses the token.
2. **Fewer launch gaps**: about 1,800 launches per token without HIP graphs.
3. **Broken pair is not fatal**: a lost peer fails every later request while
   `/ready` stays green (like upstream #278). Exiting would let a supervisor
   restart both ranks.
4. **Q8 batched MTP at C8** gains nothing over single-token batches (98.9
   against 99.7 tok/s); Q4 gains 9% there.
5. **Benchmark driver**: `model-bench.py` runs TP2 at C1 only; the concurrency
   figures come from HTTP load scripts.
6. **Q8 against an official reference**: Gufo and llama.cpp differ by KL 0.09
   on Q4 and Q8 alike; which is closer to the original model is open.
7. **Transports**: RoCEv2 mainly needs the right GID index; Thunderbolt RDMA
   (OdinLink) sends only two-sided work requests and needs a two-sided
   exchange behind `Communicator`.

## Not verified

- Cache eviction under TP2 only in the hosted test (the budget follows host
  memory and has no switch).
- The disk cache on hardware only through restart and restore; rank 1's
  eviction and a missing or corrupt rank-1 half only in the hosted tests.
- The other models' `external-model` tests need weights the two hosts lack
  (Qwen3.8 27B, DeepSeek V4 Flash); the shared serving changes are covered for
  them by the CPU scheduler, runner and cache tests.

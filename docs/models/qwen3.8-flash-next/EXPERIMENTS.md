# Qwen3.8 Flash-Next experiments

| Experiment | Decision / evidence |
| --- | --- |
| TP2 staged RDMA-read all-reduce | Retained after two-host probe and exploratory C1 AR/MTP checks: replaced the per-chunk remote-write/data-ready exchange with staged RDMA reads, an independent result window, and TCP_NODELAY. AR TP2 moved 21.19→22.23 tok/s and MTP 29.89→30.25 tok/s; completion hashes stayed unchanged. These are development measurements, not published benchmark cells. |
| TP2 bounded CQ completion spin | Rejected: a 5 µs spin increased empty CQ polls from 5,728 to about 663,000 per rank without reducing the roughly 286 ms of 50 µs sleep time; retained the existing bounded completion wait. |
| TP2 RDMA completion channel | Retained after correcting event consumption order: the two-host probe passed, and exploratory C1 AR/MTP reached 24.61/33.55 tok/s versus same-run TP1 controls of 26.58/31.34 tok/s; hashes and MTP acceptance stayed stable. The path falls back to bounded polling when the provider cannot create a completion channel. These are development measurements, not published benchmark cells. |
| Experimental TP2 official-driver overlay | Retained as a qualification tool: clean-source JSON/non-streaming d0 runs at 4cb0727 measured AR 25.06 tok/s, mixed MTP 34.17 tok/s, and repetitive MTP 39.88 tok/s; rank fingerprints, completion hashes, and MTP acceptance were retained. Artifacts are topology-isolated and the driver rejects loading, memory, image, C>1 concurrent, and nonzero-depth tables. These are not published benchmark cells. |
| Matched JSON TP1/TP2 control and C1 corpus | Qualification-only: three-repetition container controls measured TP1/TP2 AR at 26.34/24.42 tok/s and MTP at 24.51/32.77 tok/s; TP1 MTP variance was high. TP2 C1 multi measured AR 24.91 tok/s, mixed MTP 33.65 tok/s, and repetitive MTP 40.02 tok/s with forced uncached requests. A forced rank-1 removal produced the expected request failure and left no local or remote container. These are not published benchmark cells. |
| TP2 symmetric live-prefix handoff | Retained as an explicit cache-enabled experiment: control telemetry and rank-local live frontiers passed AR/MTP prefix→assistant→continuation probes; clean-source d4096 AR reused 4095 tokens at 24.54 tok/s, and MTP reached 32.57/39.49 tok/s mixed/repetitive. Explicit `cache_prompt=false` returned both ranks to a cold frontier. No snapshot bytes are transferred; default C1 remains uncached and these are not published cells. |
| TP2 rank-local immutable snapshot boundary | Retained behind the same explicit cache flag: protocol v3 handshake validates symmetric policy, both ranks retain the stable pre-generation prompt boundary in host memory, and a live continuation preserves that older snapshot for a divergent branch. Paired AR and MTP branch probes restored 671/673 cached tokens, respectively, with matching rank telemetry; MTP retained draft acceptance. Only one historical boundary is supported, and no snapshot bytes cross hosts. |
| TP2 post-v3 clean d4096 cache run | Fresh synchronized-source repetitions measured AR 24.42±0.10 tok/s, mixed MTP 31.68±2.08 tok/s, and repetitive MTP 39.07±0.59 tok/s. Cached prompt counts were 4095 for AR and 4096 for MTP; completion hashes and draft counts were retained in the isolated artifacts. These are experimental post-v3 measurements, not published cells. |
| TP2 post-v3 depth refresh | Fresh cache-enabled depth cells measured AR 21.33/20.95 tok/s and mixed MTP 30.97/32.00 tok/s at d512/d16384; repetitive MTP measured 40.34/39.31 tok/s. Cached counts ranged from 512/513 to 16357–16364, with hashes and draft counts retained. These are experimental observations, not published cells. |
| TP2 response-broker C1 smoke | After the correlated rank-0 response reader was added at `dc3198b`, fresh paired Q4 AR d0 measured 23.12 tok/s; MTP d0 measured 34.05 tok/s at 71.62% acceptance and repetitive MTP 37.01 tok/s at 100%. A cache-enabled AR d4096/context8192 probe reused 4095 tokens at 25.05 tok/s. Hashes and telemetry are in isolated `/tmp/opencode/broker-smoke-*` artifacts; these are not published cells. |
| TP2 existing long-trace profile | The retained paired ROCm trace is GPU-busy 54.1%/61.8% by rank, with dense quantized-vector kernels taking about 40% of kernel time and the largest idle gaps at host/launch boundaries. This supports profiling the staging/launch boundary before changing arithmetic; it is not a cache-enabled headline timing result. |
| Full-width MTP RMSNorm and split projection | Retained after independent CPU stage audit; one 10240-wide normalization, embedding projection shared across HC branches. |
| Full Q8 vocabulary head | Retained; private Q4 shortlist removed. Sampled top-64 proposals use exact target verification. |
| Batched MTP transformer and heads | Retained; independent body/head comparisons, private KV/recurrent/rollback/RNG state. |
| Batched decode mixers, residual epilogues and MTP norms | Retained; each request keeps its scalar reduction and private state; exact C2/C4/C6/C8 logits, acceptance and RNG. |
| HC Q8 weight prefetch | Retained for 1–8 rows of the 320×10240 projection; exact original products/FMA order, no extra allocation. |
| Q4 shared-expert weight reuse | Retained with a separate compact kernel for single-request experts; faster repetitive C1/C4/C8, no mixed-work regression, exact projection/model replay. |
| Q5 high-bit expansion | Retained; exact integer multiply/mask replaces repeated shifts, with unchanged dot products and faster serving. |
| Short convolution/history fusion | Retained for 1–8 tokens; exact output, rolling state and rollback snapshots, fewer launches and no additional allocation. |
| Batched GDN recurrence | Retained; private ragged state/rollback rows, cancellation isolation and exact session replay. Helps shallow batches most; end-to-end gains are modest. |
| Batched small projections and MoE preparation | Retained; group independent rows, quantize activations once in existing scratch, and batch router/shared-expert work. Exact scalar/batch outputs and sampled state; C1 unchanged. |
| Q4 expert grouping across the full batch | Retained; bounded groups share weights across request boundaries, with original scalar arithmetic and exact session replay. Q5 down grouping/reordering was slower on mixed routing and was rejected. |
| Expert-ordered prefill intermediates | Rejected: exact gate/up → down results with captured routing at 1024/2048 tokens, but no useful projection-chain gain for Q4_K/Q5_1 or Q5_K/Q8_0. |
| Wave64 batched Q4 gate/up | Retained above eight rows; independent 32-lane reductions preserve exact outputs and sampled state. Repetitive serving improves, C1 remains stable. Four output rows, 16-input groups and replacing the slot scan did not improve performance. |
| Register-cached vision softmax | Retained for bounded row sizes; unchanged reduction order, byte-identical Flash-Next/Qwen27B embeddings, no additional allocation. |
| 4096-patch vision attention tiles | Retained; byte-identical GEMMs/embeddings, lower latency and 24 MiB less attention scratch. Other shapes keep their original tiles. |
| Partitioned BF16 WMMA vision value projection | Rejected: failed the full-encoder reference gate despite lower isolated FP64 error. |
| Integer WMMA Q8 verification | Retained through 48 input rows for wide target projections/heads; preserves K8 partials/FMA order and reduces each result once, with exact session replay. Small batches retain vector kernels. |
| Q8 activation reuse across output rows | Rejected: no repeatable gain on the real projection shapes. |
| Wider Q8 decode and Q5 expert tiles | Rejected: 56–64 dense rows, phased/packed token tiles, Q5 wave64 and 16/32 Q5 expert rows were slower despite exact output. |
| Compact expert launch groups | Rejected: improved shared routing but negligible mixed-routing gain. |
| Sparse attention register/cache retuning | Rejected: exact d128K output, but register scheduling/occupancy gave no material gain and reloading queries was slower. |
| Shared attention selection lists | Rejected: exact outputs, but roughly 1% isolated gain did not justify another buffer and setup kernel. |
| Skip unused attention Q8 staging | Rejected: exact chunk/full-logit checks passed, but no clear end-to-end prefill gain justified the extra dispatch logic. |
| FP32 selector load scheduling | Retained; bounded scheduling removes scalar-register spills, preserves every score bit and lowers d32K selection time to 22.4 ms per pp2048. Matched d128K AR prefill improves about 1.5%. |
| Integer WMMA value transpose and paired FP32 selector lanes | Rejected: bit-preserving transpose and exact selector scores, but both were slower on deep-context inputs. |
| Packed Q8 prefill staging | Rejected: exact output, but extra decode/register/transpose costs outweighed reduced LDS use. |
| Persistent packed Q8 SSM weights | Rejected: small prefill/1–4-row gains would cost roughly 14% on eight-row decoding; two-step prefetch did not recover it. No second weight copy or private format retained. |
| Transient F16 SSM weights and parallel HC branches | Rejected: F16 staging was exact but slower overall; parallel HC branches changed quantization ties. |
| Smaller-LDS SSM projection, unrolled HC expert sum and wave64 HC combine | Rejected: exact output but no useful prefill speed gain. |
| HC quantized-output store layouts | Rejected: row-major staging plus transpose is slower; cooperative stores within one kernel save only about 2% in isolation. Residuals, normalized activations and Q8 bytes remain exact, including ragged rows. |
| Transient key transpose and mixed expert tiles | Rejected: exact outputs; key transpose slows deep selection, mixed tile sizes provide no useful prefill gain. |
| Query sharing, paired-key FP32 scoring, query LDS caching, MoE prefetch barriers/unrolling | Rejected: exact outputs, but no useful speed gain. Four-wave Q8 matrix reduction also lost to eight waves. |
| Approximate selector screening followed by exact rescoring | Rejected: GPU thresholding and compaction erased the isolated gain, before accounting for runtime error bounds. Remains a synthetic experiment; no approximate production selector added. |
| Ratio-four predictor QSA, FP32 ranking queries | Retained with sparse state, rewind and deep selector checks. |
| Greedy batch cost controller | Retained only for all-greedy C>1; separate occupancy/context bins, stable plain controls, no transition timings. Sampled replay uses fixed curves calibrated from median warmed cycles on 2026-09-20. |
| One-row MTP prefill lag | Retained; 320 KiB kept hidden state/session, avoids replaying a final prefill chunk. |
| Final-tile MTP catch-up | Retained; preserve every KV/indexer row, rank only the final attention tile, then restrict output projection, mixers and shared experts to the final aligned tile(s), with one routed Q8 expert row. Full predictor/catch-up stages, candidates and recursive carry match exactly at 224/257/2047/2048 rows. |
| Batched MTP catch-up tail | Retained; after writing every attention cache row, compact final request rows into existing scratch and skip unused output projections/FFNs. Preserve the scalar arithmetic of one-row tails; full-head candidates and recursive carry match independent execution. |
| Routed Q8 accumulation order | Retained; explicit rounded product/FMA prevents identical rows changing with column placement. Independent FP64 dot and predictor carry checks pass. |
| Isolated MMQ quantizer rounding change | Deferred: it changes half-integer tie behavior shared with W8A8 and fails their existing agreement gate. No quantizer change retained. |
| SSM tile, wave and compiler scheduling variants | Rejected: four/sixteen-wave groups, wave64, wider token tiles and iterative ILP scheduling retained exact output but did not beat the existing eight-wave projection. |
| SSM fragment lifetime and prefetch pipeline | Rejected: shorter-lived K16 fragments reduce register use without a useful gain; delayed prefetch and double-buffered LDS are slower. Projection/convolution outputs remain exact at 2048/2049 tokens. |
| Captured shared stages in batched target decoding | Rejected: exact C2/C4/C6/C8 logits, sampled state and cancellation checks, but no serving or warmed-cycle gain justified graph metadata/capture overhead. |
| Compact recurrent rollback | Retained; one full state plus exact FP32 update operands, with the original product/FMA order. All rollback prefixes match fresh execution. Depth grows on demand; seven-draft cap about 147 MiB/session, released on reset. |
| Live final frontier and async prompt snapshots | Retained; immutable branch snapshot, worker capture, bounded persistence outside the lookup lock. |
| Thread-local decode graph capture | Retained; independent snapshot workers no longer invalidate a peer's capture. Snapshot bytes and captured/replayed logits are exact; no request serialization added. |
| Chunk-equivalent projections/attention | Retained with exact continued-image/cache/full-logit gates; one-token tails keep prefill arithmetic. |
| More Q8 vocabulary rows/block | Rejected: no C1 improvement. |
| Private shared-expert F16 rows | Retained; the shared expert's SwiGLU rows for its F16 down projection get their own buffer instead of overwriting the routed experts' narrowed token rows, which removes one full-batch narrowing pass per layer. Every GEMM reads identical bytes; greedy output, chunk-boundary logits and the C2–C8 batch checks are exact. Interleaved Nix A/B at d0: 1559.6 → 1568.1 tok/s (+0.6%); the profiled kernel time fell 1362.8 → 1333.1 ms. |
| 256-row routed down-projection blocks | Rejected: exact output, Q5_1 unchanged and Q8_0 slower. The routed kernels already stream expert weights at roughly two thirds of DRAM bandwidth, so per-block prologue/epilogue and activation restaging were not the bound. |
| 64-token HC down-projection tiles | Rejected: exact output, 23% slower than the 128-token tile despite twice the resident blocks. |
| One-tile-ahead LDS fragment prefetch (dense and routed WMMA) | Rejected: exact output, but the explicit double buffer raised register use (dense 221 → 253 VGPRs; the 128-token pair reached 256 with spills) and every GEMM slowed 3–14%. |
| hipBLASLt F16 dense projections | Rejected: the pinned library reaches 19–26 TFLOPS on the 2048-token dense shapes against 30–34 TFLOPS for the Q8→F16 WMMA kernels; `tools/qwen-flash/dense_blaslt_sweep.hip` reproduces the sweep. |
| Side-stream inject/shared-expert overlap | Rejected: exact output and real kernel overlap in the trace, but the co-running kernels slowed each other and interleaved wall-clock runs were 0.5–0.9% slower. |

Separate d32K pp2048 profiling attributes 29.1% of kernel time to MoE, 34.9%
to dense projections and 12.6% to attention/indexing. Final-tile catch-up
takes 20.9 ms of MTP kernel time; the target takes 1450.2 ms.

A d0 pp2048 profile (2026-09-21) is GPU-bound: 1362.8 ms of kernel time in a
1380 ms span. Routed expert GEMMs take 35%, dense F16 projections 29%,
hyper-connection combines 10.5%, the GDN recurrence 9.5%, HC down/inject/shared
projections 7% and attention 5.5%. The dense projections run at 30–34 TFLOPS
against the measured 59 TFLOPS matrix ceiling and the combines at about
200 GB/s; the routed gate/up and down kernels stream 0.9–1.1 GB of expert
weights per layer at 160–190 GB/s.

A C4 mixed MTP trace with 32 output tokens per request is 84.9% GPU-busy
during inference, excluding model loading. Batched GDN updates take 40.9 ms,
rollback replay 14.5 ms and lazy allocations 31.8 ms. The scheduling thread
spends 5.4% of the window outside HIP API calls, including model-side CPU
work; this is not pure scheduler overhead. These are profile observations,
not unprofiled throughput measurements.

Next: improve prefill at depth and target/draft batch projection reuse while
preserving [quality](QUALITY.md). The 1700 tok/s PP and flat d0–d128K
objectives remain unmet; see [current benchmarks](BENCHMARKS.md).

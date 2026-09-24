# TP=2 RDMA development probe

This is the first Qwen3.8-Flash-Next two-rank execution path. It is a
qualification probe, not yet the HTTP serving topology.

## Build

The optional adapter uses the pinned `rdma-core`/`libibverbs` package and is
disabled in the default build.

```sh
nix develop --inputs-from .#tp2-rdma -c cmake --preset gpu-tp2
nix develop --inputs-from .#tp2-rdma -c cmake --build --preset gpu-tp2 \
  --target qwen38_flash_next_gpu_probe
```

The target and MTP sidecar must be present on both hosts. The first run uses
the verified `UD-Q4_K_XL` target and shared-Q8_0 MTP sidecar.

## Two-host probe

Run rank 0 first. `BOOTSTRAP_HOST` is the address rank 1 uses to reach rank
0's TCP metadata endpoint; model data itself uses the native IB QP.

```sh
# host 0
build/gpu-tp2/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mtp-model models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --prompt 'The capital of France is' --batch 32 --context 4096 \
  --tp-world-size 2 --tp-rank 0 --tp-bootstrap-port 18515

# host 1
build/gpu-tp2/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mtp-model models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --prompt 'The capital of France is' --batch 32 --context 4096 \
  --tp-world-size 2 --tp-rank 1 --tp-bootstrap-host RANK0_ADDRESS \
  --tp-bootstrap-port 18515
```

The loader discovers the remaining target shards beside the first shard.

## Current boundaries

- Expert-parallel routed MoE with replicated dense, attention, GDN, HC,
  embedding, and LM-head weights.
- Rank 0 owns the shared expert; the routed layer output is reduced with the
  ordered communicator.
- Host-staged synchronous RDMA is used first; HIP graph capture is disabled.
- Distributed snapshots, disk continuation, and vision are rejected for now.
- MTP follows the same expert partition and collective path.
- Q8_0 uses the same partition/upload contract; its larger target must pass a
  local-memory fit check before it is promoted to a serving target.

# ComfyUI-SolAttn_triton-video

Video-oriented Sol-Attn Triton nodes for long Wan/MiniMax-H3/SCAIL2 sequences.

This package is separate from upstream `ComfyUI-SolAttn_triton` and exposes
`SolAttnVideoPatch` (displayed as **Patch Sol-Attn Video**) and
`SolAttnVideoBlockProbe` (displayed as **Sol-Attn Video Block Probe**). On RTX
4090, `SageAttentionVideoSM89Patch` provides an exact dense four-tile Sage path
for quality-first Wan/SCAIL2 workloads.

The production SCAIL2 chain is self-contained at the ComfyUI-node level:

- `SageAttentionVideoSM89Patch` — exact dense SM89 Four-Tile attention;
- `SCAIL2VideoFinalPoseQueryPrune` — keeps complete pose K/V and omits only
  final pose queries that SCAIL2 discards;
- `SolAttnVideoSoftCleanup` — releases allocator cache after saving without
  unloading the warm model.

The SCAIL2 production workflow does **not** require the experimental
`ComfyUI-SCAIL2-KitchenW4A4` package. It does require the separately compiled
`h3_sage_sm89_backend` Python/CUDA extension. Installing this repository alone
does not install that binary backend.

## RTX 4090 production requirements

- NVIDIA SM89 GPU (the accepted target is RTX 4090);
- the exact PyTorch/CUDA ABI used to build `h3_sage_sm89_backend`;
- Triton compatible with that PyTorch installation;
- a persistent writable Triton and CUDA cache directory;
- the `h3_sage_sm89_backend` wheel built for the production Python, PyTorch,
  CUDA and Linux ABI.

The validated machine currently uses Python 3.12, PyTorch `2.9.1+cu130`,
Triton `3.5.1`, and an SM89-only backend build. Do not copy the compiled `.so`
to a machine with a different Python/PyTorch/CUDA ABI. Build or install a
matching wheel instead.

For production upgrades, deploy a clean tagged checkout beside the active
directory, install the matching backend wheel with ComfyUI's own Python, start
ComfyUI, verify all three node IDs through `/object_info`, and run a short
same-seed smoke test before switching traffic. Do not run `git pull` over a
dirty active custom-node directory.

Recommended persistent cache variables:

```bash
export TRITON_CACHE_DIR=/path/to/persistent-cache/triton
export TORCH_EXTENSIONS_DIR=/path/to/persistent-cache/torch-extensions
export CUDA_CACHE_PATH=/path/to/persistent-cache/cuda
```

The Four-Tile backend is an ahead-of-time compiled CUDA extension; Triton cache
persistence mainly benefits the Sol kernels and their first-run autotuning.

It contains long-sequence routing, optional Morton frame ordering, video/audio
conditioning safeguards, tri-policy Sol/Sage dispatch, and distant INT8 QK/PV.

## Example workflow

- [`SCAIL2_720P_accelerated.json`](example_workflows/SCAIL2_720P_accelerated.json) —
  SCAIL2 720P video workflow using Sage/Sol layer dispatch and the warm-model
  soft VRAM cleanup node.

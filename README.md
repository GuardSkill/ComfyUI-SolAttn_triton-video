# ComfyUI-SolAttn_triton-video

中文文档：

- [RTX 4090 中文安装指南](docs/INSTALL_RTX4090_ZH.md)
- [SCAIL2 加速实现与测试记录](docs/SCAIL2_81x76_ACCELERATION_ZH.md)

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
`h3_sage_sm89_backend` Python/CUDA extension. A validated RTX 4090 binary wheel
is included under [`wheels/`](wheels/); it must be installed explicitly with
ComfyUI's Python.

## Installation / 安装

The prebuilt backend targets Linux x86_64, RTX 4090/SM89, PyTorch
`2.9.1+cu130`, Triton `3.5.1`, and Python 3.10–3.12. It contains the same
optimized kernel revision for all three Python ABIs.

| ComfyUI Python | Bundled wheel |
|---|---|
| 3.10 | `h3_sage_sm89_backend-0.1.0-cp310-cp310-linux_x86_64.whl` |
| 3.11 | `h3_sage_sm89_backend-0.1.0-cp311-cp311-linux_x86_64.whl` |
| 3.12 | `h3_sage_sm89_backend-0.1.0-cp312-cp312-linux_x86_64.whl` |

Clone the node into `custom_nodes`, then install the wheel selected from the
Python interpreter that actually starts ComfyUI:

```bash
COMFY_ROOT=/path/to/ComfyUI
COMFY_PYTHON=/path/to/comfyui/python
NODE_ROOT="$COMFY_ROOT/custom_nodes/ComfyUI-SolAttn_triton-video"

git clone https://github.com/GuardSkill/ComfyUI-SolAttn_triton-video.git \
  "$NODE_ROOT"

PY_TAG=$("$COMFY_PYTHON" -c \
  'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
case "$PY_TAG" in cp310|cp311|cp312) ;; *)
  echo "No bundled wheel for $PY_TAG" >&2; exit 1;;
esac

BACKEND_WHEEL="$NODE_ROOT/wheels/h3_sage_sm89_backend-0.1.0-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
(cd "$NODE_ROOT/wheels" && sha256sum -c SHA256SUMS)
"$COMFY_PYTHON" -m pip install --no-deps --force-reinstall "$BACKEND_WHEEL"
```

Do not run the install command with a system `pip`; use ComfyUI's Python.
`--no-deps` intentionally prevents the wheel from replacing the working
PyTorch/CUDA stack. SageAttention must already be installed in that same
environment.

Restart ComfyUI completely, then verify the binary backend:

```bash
"$COMFY_PYTHON" - <<'PY'
import torch, triton, h3_sage_sm89_backend
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("triton:", triton.__version__)
print("gpu:", torch.cuda.get_device_name())
print("capability:", torch.cuda.get_device_capability())
assert torch.cuda.get_device_capability() == (8, 9)
assert callable(h3_sage_sm89_backend.attention)
print("SM89 backend OK")
PY
```

The production workflow should expose these nodes after restart:

- `SageAttentionVideoSM89Patch`
- `SCAIL2VideoFinalPoseQueryPrune`
- `SolAttnVideoSoftCleanup`

For a safe in-place upgrade, persistent kernel caches, `/object_info`
validation, and troubleshooting, read the
[Chinese RTX 4090 installation guide](docs/INSTALL_RTX4090_ZH.md).

## RTX 4090 production requirements

- NVIDIA SM89 GPU (the accepted target is RTX 4090);
- the exact PyTorch/CUDA ABI used to build `h3_sage_sm89_backend`;
- Triton compatible with that PyTorch installation;
- a persistent writable Triton and CUDA cache directory;
- the bundled `h3_sage_sm89_backend` wheel, or a wheel rebuilt for the target
  Python, PyTorch, CUDA and Linux ABI.

The validated wheel matrix uses Python 3.10, 3.11, and 3.12 with PyTorch
`2.9.1+cu130`, Triton `3.5.1`, and an SM89-only backend build. Do not copy the
compiled `.so` to a different Python/PyTorch/CUDA ABI. Install the matching
wheel instead.

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

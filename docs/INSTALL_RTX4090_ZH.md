# ComfyUI-SolAttn_triton-video：RTX 4090 中文安装指南

本文用于在 RTX 4090（SM89）生产机上安装经过验证的 SCAIL2 精确
Four-Tile 加速链。该链不使用 Sol 稀疏注意力，包含以下三个节点：

- `SageAttentionVideoSM89Patch`：全层精确 dense Four-Tile Sage；
- `SCAIL2VideoFinalPoseQueryPrune`：保留完整 pose K/V，只省略模型最终会丢弃的 pose query；
- `SolAttnVideoSoftCleanup`：任务结束后清理 allocator cache，但不卸载热模型。

最终工作流不再依赖实验性的 `ComfyUI-SCAIL2-KitchenW4A4`。完整 UI
仍可能使用 KJNodes、Easy-Use、VideoHelperSuite、rgthree 等通用节点包。

## 1. 安装前检查

先找到 ComfyUI 实际使用的 Python。不要直接使用系统 `pip`。

```bash
COMFY_ROOT=/root/lisiyuan/ComfyUI
COMFY_PYTHON=/root/lisiyuan/miniforge3/envs/comfyui/bin/python

"$COMFY_PYTHON" - <<'PY'
import sys, torch, triton
print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("triton:", triton.__version__)
assert torch.cuda.is_available()
print("gpu:", torch.cuda.get_device_name())
print("capability:", torch.cuda.get_device_capability())
assert torch.cuda.get_device_capability() == (8, 9), "Four-Tile 生产 kernel 仅支持 SM89"
PY

nvcc --version
```

已验证的生产环境是 Python 3.12、PyTorch `2.9.1+cu130`、Triton
`3.5.1` 和 RTX 4090。其他版本不代表一定不能运行，但 CUDA 扩展必须按
目标机的 Python、PyTorch、CUDA ABI 重新构建，不能跨环境复制 `.so`。

## 2. 安装 ComfyUI 节点

新安装：

```bash
COMFY_ROOT=/root/lisiyuan/ComfyUI
git clone https://github.com/GuardSkill/ComfyUI-SolAttn_triton-video.git \
  "$COMFY_ROOT/custom_nodes/ComfyUI-SolAttn_triton-video"
git -C "$COMFY_ROOT/custom_nodes/ComfyUI-SolAttn_triton-video" \
  checkout e35e4e5
```

更新现有安装前，必须先确认目录干净并停止提交新任务：

```bash
COMFY_ROOT=/root/lisiyuan/ComfyUI
NODE_ROOT="$COMFY_ROOT/custom_nodes/ComfyUI-SolAttn_triton-video"

git -C "$NODE_ROOT" status --short
git -C "$NODE_ROOT" fetch origin
git -C "$NODE_ROOT" checkout e35e4e5
```

如果 `git status --short` 有输出，不要直接 `pull` 或强制覆盖。应先把当前目录
整体移动到 `custom_nodes` 之外的回滚目录，再进行干净克隆。回滚副本不能留在
`custom_nodes` 内，否则 ComfyUI 会扫描两份插件并产生重复节点注册。

## 3. 安装 SM89 CUDA backend

Four-Tile 依赖 `h3_sage_sm89_backend`。它是 Python/CUDA 二进制扩展，不是
ComfyUI 节点，也不会随本仓库自动安装。

### 推荐：安装匹配环境的 wheel

从私有制品库取得与目标环境完全匹配的 wheel，然后使用 ComfyUI Python 安装：

```bash
COMFY_PYTHON=/root/lisiyuan/miniforge3/envs/comfyui/bin/python
BACKEND_WHEEL=/绝对路径/h3_sage_sm89_backend-0.1.0-cp312-cp312-linux_x86_64.whl

"$COMFY_PYTHON" -m pip install --no-deps --force-reinstall "$BACKEND_WHEEL"
```

文件名只能提示 Python ABI；仍需确认该 wheel 对应目标 PyTorch 和 CUDA 版本。
公开仓库不包含核心 CUDA 源码，这是为了避免公开生产 kernel 的实现细节。

### 有授权源码时本机编译

```bash
COMFY_PYTHON=/root/lisiyuan/miniforge3/envs/comfyui/bin/python
BACKEND_SOURCE=/绝对路径/h3_sage_sm89_backend

cd "$BACKEND_SOURCE"
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=8 \
  "$COMFY_PYTHON" -m pip install . --no-build-isolation --no-deps
```

不要在安装 backend 时顺便升级 PyTorch、CUDA 或 Triton。生产环境的 ABI 发生变化
后必须重新构建 wheel，并重新执行速度和质量回归。

## 4. 配置持久化编译缓存

将下面的变量写入启动 ComfyUI 的脚本或 systemd 环境，而不是只在临时终端执行：

```bash
COMFY_ROOT=/root/lisiyuan/ComfyUI
export TRITON_CACHE_DIR="$COMFY_ROOT/.kernel_cache/triton"
export TORCH_EXTENSIONS_DIR="$COMFY_ROOT/.kernel_cache/torch_extensions"
export CUDA_CACHE_PATH="$COMFY_ROOT/.kernel_cache/cuda"

mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH"
```

Four-Tile 主 kernel 是预编译 CUDA 扩展。持久化 Triton cache 主要改善 Sol/Triton
算子的冷启动与 autotune，不会让已经编译好的 Four-Tile `.so` 再次变快。不要在每个
任务结束后删除这些目录。

## 5. 重启并验证

完全重启 ComfyUI 后，先检查 backend：

```bash
COMFY_PYTHON=/root/lisiyuan/miniforge3/envs/comfyui/bin/python

"$COMFY_PYTHON" - <<'PY'
import inspect, torch, h3_sage_sm89_backend
print("backend:", inspect.getfile(h3_sage_sm89_backend))
print("gpu capability:", torch.cuda.get_device_capability())
assert torch.cuda.get_device_capability() == (8, 9)
PY
```

ComfyUI 启动后检查三个生产节点：

```bash
curl -fsS http://127.0.0.1:8188/object_info > /tmp/comfy_object_info.json
COMFY_PYTHON=/root/lisiyuan/miniforge3/envs/comfyui/bin/python

"$COMFY_PYTHON" - <<'PY'
import json
info = json.load(open("/tmp/comfy_object_info.json", encoding="utf-8"))
required = (
    "SageAttentionVideoSM89Patch",
    "SCAIL2VideoFinalPoseQueryPrune",
    "SolAttnVideoSoftCleanup",
)
missing = [name for name in required if name not in info]
assert not missing, f"缺少节点: {missing}"
print("生产加速节点注册正常")
PY
```

最后先运行短时长、较低分辨率的同 seed 冒烟测试，再运行 920K/12 秒回归。日志中
应出现：

- `enabled exact dense SM89 four-tile Sage`；
- `installed exact SCAIL2 final pose-query pruning`；
- 末尾清理节点的显存统计。

该质量版工作流不应连接 `SolAttnVideoPatch`。如果日志显示 sparse/Sol 路由正在执行，
说明导入了错误工作流或模型链连线不正确。

## 6. 常见问题

### 提示 backend 未安装

确认 wheel 是用 ComfyUI 的 Python 安装：

```bash
/root/lisiyuan/miniforge3/envs/comfyui/bin/python -m pip show h3-sage-sm89-backend
```

### `undefined symbol` 或导入 `.so` 失败

通常是 Python、PyTorch 或 CUDA ABI 不匹配。卸载旧 wheel，按当前环境重新构建，
不要只从另一台机器复制 `.so`。

### 首次运行慢

Triton autotune 和 CUDA cache 的首次建立属于正常现象。保持 cache 目录不变，再比较
第二次热运行。Four-Tile CUDA 扩展本身不应在每次任务中重新编译。

### ComfyUI 出现重复节点

检查 `custom_nodes` 下是否同时存在正式目录、备份目录或旧名称副本。回滚目录必须放在
`custom_nodes` 之外。

### 是否需要 KitchenW4A4

本工作流不需要。旧 `ComfyUI-SCAIL2-KitchenW4A4` 可以保留用于其他实验工作流，但
不能把旧的 `SCAIL2FinalPoseQueryPrune` 节点重新接入本工作流。

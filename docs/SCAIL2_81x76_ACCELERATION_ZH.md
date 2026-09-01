# SCAIL2 920K 视频加速与残影控制（RTX 4090）

本文记录 `ComfyUI-SolAttn_triton-video` 在 SCAIL2 人物动作迁移工作流中的实现方式、测试结论和推荐参数。目标是在 RTX 4090 24GB 上保留可用画质，同时缩短 920K 像素、6 秒、150 帧视频的推理时间。

## 结论

当前推荐方案不是单独打开一个 INT8 开关，而是同时修正两层问题：

1. 使用视频专用节点 `SolAttnVideoPatch`，让中间 24 个 transformer block 使用 Sol 稀疏注意力与远距离 INT8 QK/PV；首尾 16 个 block 保持已有的全覆盖注意力路径。
2. 按 SCAIL2 的训练方式使用 81 帧窗口、76 帧步长和 5 帧重叠，最后把循环产生的尾部填充裁剪为原视频的 150 帧。

同一台 RTX 4090 上，推荐的 `81/76 + tau=1.5` 工作流耗时为 **196.63 秒**。此前 125 帧分段的参考运行约为 **211.19 秒**，本次组合优化缩短约 **6.9%**。更重要的是，原工作流在第 124 帧附近的大幅跳变明显降低。

这 6.9% 是端到端组合收益，包含更合适的分段调度和 Sol kernel，不应表述为单个 Triton kernel 的独立加速比。

## 为什么 125 帧分段容易出现残影

ComfyUI 的 `WanSCAILToVideo` 节点说明中明确记录：SCAIL2 训练使用 81 帧 chunk、76 帧 step，也就是下一段复用上一段末尾 5 帧作为 anchor。

原测试工作流使用 125 帧 chunk，并在 150 帧视频末尾生成第二段。最大时序跳变恰好出现在第 124 帧附近。原生/高质量注意力路径也可能出现这个现象，INT8 和稀疏路由会放大已有的不连续，但不是唯一根因。

正确调度为：

```text
第 1 段：输入窗口 0..80       （81 帧）
第 2 段：从第 76 帧继续       （复用 5 帧）
最终输出：裁剪到输入视频长度   （本例 150 帧）
```

不要简单把 chunk 从 125 改成 81 后仍按 81 前进；那样没有 5 帧重叠。工作流中必须分别提供 `chunk_size=81` 和 `stride=76`。

## `SolAttnVideoPatch` 做了什么

`SolAttnVideoPatch` 是本仓库的视频专用自定义节点，不是上游节点的简单改名。当前路径包含以下策略：

- transformer blocks `8-31`：使用 Sol-Attn 稀疏路由。
- blocks `0-7,32-39`：交给已有的全覆盖注意力路径。首尾层的误差更容易直接传播到最终输出，因此不做 Sol 稀疏化。
- `int8_qk=true`：QK 精确分支使用 INT8 kernel。该开关决定是否进入 INT8 Sol kernel 主路径。
- `int8_pv=true`：P@V 同样使用 INT8，继续降低远距离分支的带宽与计算成本。
- `int8_scope=distant`：conditioning、局部邻域及受保护块保持精确计算，仅远距离路由块使用 INT8。
- `local_blocks=2`：每个 query block 周围保留两个精确局部 block。
- `sink_conditioning=exact_kv_and_rows`：conditioning KV 与 conditioning query rows 保持精确，降低身份、参考图和控制信息漂移。
- `use_tma=false`：RTX 4090 是 SM89；测试中 TMA 路径没有收益。
- `morton=false`、`coordinate_routing=false`：这是本次 SCAIL2 实测采用的配置，不把尚未通过该工作负载验证的重排策略混入默认方案。

## 推荐参数

### 平衡档（默认）

```text
backend                 = sol_sparse
tau                     = 1.5
start_percent           = 0.0
end_percent             = 1.0
min_tokens              = 4096
int8_qk                 = true
int8_pv                 = true
int8_scope              = distant
local_blocks            = 2
sink_conditioning       = exact_kv_and_rows
dense_blocks            = 0-7,32-39
sol_blocks              = 8-31
sage_blocks             = 0-7,32-39
int8_dense_blocks       = 空
int8_start_percent      = 0.0
int8_end_percent        = 1.0
morton                   = false
coordinate_routing      = false
use_tma                 = false
```

SCAIL2 工作流参数：

```text
chunk_size              = 81
chunk_stride            = 76
previous_frame_count    = 5
output_frames           = 输入帧数（本例 150）
```

### 质量档

把 `tau` 调到 `1.2`。更多远距离 block 会被保留，速度会下降。本次 81/76 测试耗时为 248.55 秒，因此它适合问题输入的排查，不适合作为 4090 默认档。

### 极速档

把 `tau` 调到 `1.8`。本次 81/76 测试耗时为 191.43 秒。时序诊断仍明显优于错误的 125 帧分段，但稀疏程度更高；建议对手部快速运动、遮挡、多人交叉等输入先做抽样检查。

## 实测数据

工作负载：RTX 4090 24GB、720×1280（约 920K 像素）、150 帧、25 FPS、6 秒、4 个采样步、固定 seed。

| 配置 | 端到端耗时 | 平均相邻帧灰度 MAE | 诊断帧 124 MAE | 说明 |
|---|---:|---:|---:|---|
| 125 帧参考路径 | 211.19 s | 17.172 | 63.91 | 错误分段边界附近出现大跳变 |
| 125 帧，双 INT8，tau 1.2 | 228.07 s | 16.805 | 58.98 | 降低 tau 只能部分缓解 |
| 125 帧，双 INT8，tau 0.8 | 245.23 s | 16.730 | 51.06 | 更慢，仍没有解决分段根因 |
| 81/76，双 INT8，tau 1.2 | 248.55 s | 16.129 | 7.68 | 质量诊断档，原始循环输出 152 帧 |
| 81/76，双 INT8，tau 1.8 | 191.43 s | 16.880 | 8.83 | 极速档，原始循环输出 152 帧 |
| **81/76，双 INT8，tau 1.5** | **196.63 s** | **16.230** | **10.03** | **推荐档，已裁剪为 150 帧** |

“相邻帧灰度 MAE”只用于发现突然闪变和分段不连续，不是感知画质分数。改动分段后，视频中的动作相位也可能变化，因此不同配置的单帧 MAE 不代表逐像素内容相同。最终结论仍需结合完整视频、手脸局部和多个输入进行人工检查。

## 已排除或不推荐的方案

### 只关闭 INT8 PV

保持 QK INT8、关闭 PV INT8 的测试耗时为 223.26 秒，异常位置与双 INT8 路径基本一致。说明残影不是 PV 量化单独造成；关闭 PV 会损失速度，却没有解决分段问题。

### 只扩大局部窗口

把 `local_blocks` 从 2 增加到 3 的双 INT8 测试耗时为 220.44 秒，诊断帧跳变没有改善。局部窗口不是第 124 帧分段残影的根因。

### 静态关闭少数 block 或最后一步回退 BF16

关闭 block 8、31 的 INT8，并让最后一个 denoise step 回退 BF16，耗时为 263.68 秒，诊断帧反而更差。静态选择几个层不能替代正确的 chunk overlap。

### 单个 150 帧 chunk

920K 像素下，150 帧单 chunk 在 RTX 4090 24GB 上曾于 FP8 activation 临时分配处 OOM。不要为了绕过分段直接使用 150 帧窗口；81/76 是更稳定的显存与质量方案。

### 当前 W4A4 路径

comfy-kitchen/Nunchaku W4A4 的独立线性层 kernel 有加速，但现有 SCAIL2 后训练权重在完整 denoise trajectory 上仍有累计误差，端到端收益也被外部模块调度和换入抵消。因此本工作流没有把实验性 W4A4 作为公开质量档的一部分。

## 使用与测速技巧

1. 第一次遇到新 token shape 时 Triton 会 autotune；冷启动可多出约 20–50 秒。比较速度时应先预热，再比较同 shape 的 hot run。
2. 固定输入图、参考视频、seed、分辨率、帧数、LoRA、sampler 和 steps。任何一项变化都会使画质与耗时对比失真。
3. 日志中应看到类似 `sparse (...tokens..., 40, 128) tau=1.5 int8 pointer`，否则可能没有进入双 INT8 Sol 路径。
4. cross-attention 保持 dense 是正常行为；Sol 节点只接管符合条件的长序列 self-attention。
5. 发布工作流时使用 UI JSON；`{"prompt": ...}` 顶层结构是 API prompt，拖入 ComfyUI 画布不会显示节点。
6. 对不同人物和舞蹈视频至少检查：分段交界前后 10 帧、快速手势、手脸交叉、身体遮挡、画面边缘和多人交叉。
7. 如果仍有闪变，先确认 chunk/stride/overlap，再将 tau 从 1.5 降到 1.2。不要第一时间关闭全部 INT8，否则会同时失去主要加速路径。

## 文件

- 可拖入 ComfyUI 的 UI 工作流：`experiments/scail_accel/workflows/SCAIL2_920K_6S_RTX4090_SOL_81x76_TAU15_UI.json`
- API prompt：`experiments/scail_accel/workflows/SCAIL2_920K_6S_RTX4090_SOL_81x76_TAU15_API.json`
- UI 工作流生成器：`experiments/scail_accel/make_81x76_ui_workflow.py`
- 视频节点实现：`third_party/ComfyUI-SolAttn_triton-video/__init__.py`


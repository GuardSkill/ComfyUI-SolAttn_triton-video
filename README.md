# ComfyUI-SolAttn_triton-video

Video-oriented Sol-Attn Triton nodes for long Wan/MiniMax-H3/SCAIL2 sequences.

This package is separate from upstream `ComfyUI-SolAttn_triton` and exposes
`SolAttnVideoPatch` (displayed as **Patch Sol-Attn Video**) and
`SolAttnVideoBlockProbe` (displayed as **Sol-Attn Video Block Probe**). On RTX
4090, `SageAttentionVideoSM89Patch` provides an exact dense four-tile Sage path
for quality-first Wan/SCAIL2 workloads.

It contains long-sequence routing, optional Morton frame ordering, video/audio
conditioning safeguards, tri-policy Sol/Sage dispatch, and distant INT8 QK/PV.

## Example workflow

- [`SCAIL2_720P_accelerated.json`](example_workflows/SCAIL2_720P_accelerated.json) —
  SCAIL2 720P video workflow using Sage/Sol layer dispatch and the warm-model
  soft VRAM cleanup node.

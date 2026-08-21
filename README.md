# ComfyUI-SolAttn_triton-video

Video-oriented Sol-Attn Triton nodes for long Wan/MiniMax-H3/SCAIL2 sequences.

This package is separate from upstream `ComfyUI-SolAttn_triton` and exposes
`SolAttnVideoPatch` (displayed as **Patch Sol-Attn Video**) and
`SolAttnVideoBlockProbe` (displayed as **Sol-Attn Video Block Probe**).

It contains long-sequence routing, optional Morton frame ordering, video/audio
conditioning safeguards, tri-policy Sol/Sage dispatch, and distant INT8 QK/PV.

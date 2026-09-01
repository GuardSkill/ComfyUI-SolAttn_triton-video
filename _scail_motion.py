"""Motion-adaptive exact block maps for SCAIL/SCAIL2 merged attention.

SCAIL concatenates [reference + generated video] and pose tokens before dense
self-attention. Sequence-neighbour blocks therefore do not reliably represent
temporal neighbours at frame or stream boundaries. This module observes the
pose embedding input, estimates fast-motion frames, and builds a compact
Q-block/K-block exactness map consumed by the existing Sol pointer kernel.
"""

from __future__ import annotations

import logging
import weakref

import torch


class SCAILMotionProtection:
    def __init__(self, model, *, block_size: int = 64,
                 temporal_radius: int = 1, landmark_stride: int = 4):
        self.model_ref = weakref.ref(model)
        self.block_size = int(block_size)
        self.temporal_radius = int(temporal_radius)
        self.landmark_stride = int(landmark_stride)
        self.pose_grid = None
        self.high_motion = None
        self.signature = None
        self.map_cache = {}
        self._logged = set()
        pose_embed = getattr(model, "patch_embedding_pose", None)
        self.pose_attribute = "patch_embedding_pose"
        if pose_embed is None:
            pose_embed = getattr(model, "pose_patch_embedding", None)
            self.pose_attribute = "pose_patch_embedding"
        if pose_embed is None:
            raise ValueError(
                "SCAIL motion routing requires patch_embedding_pose or "
                "pose_patch_embedding"
            )
        self.hook = pose_embed.register_forward_hook(self._observe_pose)
        logging.info(
            "[sol_attn] installed SCAIL motion protection on %s",
            self.pose_attribute,
        )

    def _observe_pose(self, _module, inputs, output):
        if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
            return
        pose = inputs[0]
        if pose.ndim != 5 or output.ndim != 5:
            logging.warning(
                "[sol_attn] SCAIL pose hook expected rank-5 input/output, got %s/%s",
                tuple(pose.shape), tuple(output.shape),
            )
            self.pose_grid = None
            self.high_motion = None
            return
        signature = (pose.untyped_storage().data_ptr(), tuple(pose.shape), pose.device)
        self.pose_grid = tuple(int(x) for x in output.shape[-3:])
        if signature == self.signature:
            return
        self.signature = signature
        self.map_cache.clear()
        # Pose conditioning is constant across denoise steps. Keep this entirely
        # on GPU and cache it by storage pointer to avoid per-block CPU sync.
        sample = pose.detach().float()
        if sample.shape[-3] <= 1:
            self.high_motion = torch.zeros(
                (sample.shape[-3],), device=sample.device, dtype=torch.bool)
            return
        delta = (sample[:, :, 1:] - sample[:, :, :-1]).abs().mean(
            dim=(0, 1, 3, 4))
        # Robust relative threshold. At least the upper quartile is considered
        # fast, while a single extreme frame cannot force every frame exact.
        threshold = torch.maximum(
            delta.mean() + delta.std(unbiased=False),
            torch.quantile(delta, 0.85),
        )
        transitions = delta >= threshold
        high = torch.zeros((sample.shape[-3],), device=sample.device, dtype=torch.bool)
        high[:-1] |= transitions
        high[1:] |= transitions
        self.high_motion = high

    def __call__(self, transformer_options, tokens: int, device) -> torch.Tensor | None:
        grid = (transformer_options or {}).get("grid_sizes")
        if grid is None or self.pose_grid is None or self.high_motion is None:
            return None
        main_t, main_h, main_w = (int(x) for x in grid)
        pose_t, pose_h, pose_w = self.pose_grid
        main_frame_tokens = main_h * main_w
        pose_frame_tokens = pose_h * pose_w
        main_tokens = main_t * main_frame_tokens
        pose_tokens = pose_t * pose_frame_tokens
        if tokens != main_tokens + pose_tokens:
            return None
        key = (tokens, main_t, main_h, main_w, pose_t, pose_h, pose_w,
               device, self.signature)
        cached = self.map_cache.get(key)
        if cached is not None:
            return cached

        blocks = (tokens + self.block_size - 1) // self.block_size
        centers = torch.arange(blocks, device=device, dtype=torch.int64)
        centers = torch.minimum(
            centers * self.block_size + self.block_size // 2,
            torch.tensor(tokens - 1, device=device, dtype=torch.int64))
        is_pose = centers >= main_tokens
        main_frame = torch.div(centers, main_frame_tokens, rounding_mode="floor")
        pose_frame = torch.div(
            (centers - main_tokens).clamp_min(0), pose_frame_tokens,
            rounding_mode="floor")
        reference_frames = max(main_t - pose_t, 0)
        global_frame = torch.where(is_pose, pose_frame + reference_frames, main_frame)
        qf, kf = global_frame[:, None], global_frame[None, :]
        distance = (qf - kf).abs()

        main_position = centers % main_frame_tokens
        pose_position = (centers - main_tokens).clamp_min(0) % pose_frame_tokens
        y = torch.where(
            is_pose,
            torch.div(pose_position, pose_w, rounding_mode="floor"),
            torch.div(main_position, main_w, rounding_mode="floor"),
        )
        x = torch.where(is_pose, pose_position % pose_w, main_position % main_w)
        height = torch.where(
            is_pose, torch.tensor(pose_h, device=device),
            torch.tensor(main_h, device=device))
        width = torch.where(
            is_pose, torch.tensor(pose_w, device=device),
            torch.tensor(main_w, device=device))
        # Compare streams on a common normalized 32x32 plane. A block is a
        # one-dimensional 64-token strip, so the X tolerance is wider than Y.
        yn = torch.div(y * 32, height, rounding_mode="floor")
        xn = torch.div(x * 32, width, rounding_mode="floor")
        spatial_match = ((yn[:, None] - yn[None, :]).abs() <= 3) & \
                        ((xn[:, None] - xn[None, :]).abs() <= 7)

        motion = self.high_motion.to(device=device)
        main_motion_index = (main_frame - reference_frames).clamp(0, max(pose_t - 1, 0))
        pose_motion_index = pose_frame.clamp(0, max(pose_t - 1, 0))
        block_motion = torch.where(
            is_pose,
            motion[pose_motion_index],
            (main_frame >= reference_frames) & motion[main_motion_index],
        )
        fast_radius = self.temporal_radius + 2
        radius = torch.where(
            block_motion[:, None] | block_motion[None, :],
            torch.tensor(fast_radius, device=device),
            torch.tensor(self.temporal_radius, device=device),
        )
        protected = (distance <= radius) & spatial_match

        # Every query sees reference frames exactly. Landmark frames retain a
        # sparse spatial lattice, rather than making a whole frame dense.
        reference_key = (~is_pose) & (main_frame < reference_frames)
        protected |= reference_key[None, :]
        if self.landmark_stride > 0:
            spatial_block = torch.where(
                is_pose,
                torch.div((centers - main_tokens).clamp_min(0) % pose_frame_tokens,
                          self.block_size, rounding_mode="floor"),
                torch.div(centers % main_frame_tokens,
                          self.block_size, rounding_mode="floor"),
            )
            landmark_key = ((global_frame % self.landmark_stride) == 0) & \
                           ((spatial_block % 8) == 0)
            protected |= landmark_key[None, :]

        result = protected.to(torch.uint8).contiguous()
        self.map_cache[key] = result
        log_key = (tokens, self.pose_grid, tuple(grid))
        if log_key not in self._logged:
            self._logged.add(log_key)
            density = float(result.float().mean())
            high_count = int(self.high_motion.sum())
            logging.info(
                "[sol_attn] SCAIL motion map: main=%s pose=%s blocks=%d "
                "high_motion_frames=%d/%d exact_density=%.2f%%",
                tuple(grid), self.pose_grid, blocks, high_count, pose_t,
                density * 100.0,
            )
        return result


def install_scail_motion_protection(model, *, block_size=64,
                                    temporal_radius=1, landmark_stride=4):
    existing = getattr(model, "_sol_scail_motion_protection", None)
    if existing is not None:
        existing.temporal_radius = int(temporal_radius)
        existing.landmark_stride = int(landmark_stride)
        existing.map_cache.clear()
        return existing
    protection = SCAILMotionProtection(
        model, block_size=block_size, temporal_radius=temporal_radius,
        landmark_stride=landmark_stride)
    setattr(model, "_sol_scail_motion_protection", protection)
    return protection

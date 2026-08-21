"""Morton (Z-order) reordering of the video span for MiniMax-H3.

H3 packs one self-attention sequence: [text][cond][ref_img][ref_audio][audio][video].
Only the target video segment is reordered -- text, audio and reference rows keep
their positions, so the conditioning streams are untouched and only the video
tokens get 3D-compact attention blocks.

The video segment is contiguous, always last, and laid out t-major then h then w
(see _video_grid / unpatchify_video), so the permutation maps straight onto it.

Revisions of the model differ in how the layout is obtained -- some expose a
_layout() method, others build PackedLayout inline from minimax_payload -- so the
span is registered from PackedLayout.__init__ itself, which is common to both,
and looked up by the identity of the position_ids tensor that rope_freqs is
handed. Nothing is permuted at construction time; the gate is read per forward.

Three coordinated pieces, all gated on transformer_options["sol_morton"]:
  * _forward    -- read the gate
  * rope_freqs  -- permute the matching rows of position_ids, publish the span
  * block hooks -- permute the hidden states in, and restore them out
Because tokens and their positions are permuted together and undone before the
final layer, it is exactly neutral for dense attention.
"""

import logging
import sys

import torch

from ._morton import morton_perm


_INSTALLED = set()
_PATCHED_LAYOUTS = set()
BLOCK_SIZE = 64  # kernel block size; the permutation is aligned to this grid
_DEVICE_CACHE = {}
_ROUTE_CACHE = {}
# id(position_ids) -> (layout, span). The layout is kept alive deliberately so
# the id cannot be recycled underneath us; there is one entry per distinct shape.
_SPANS = {}


def _frame_route_map(bounds, span, radius, device, *, landmark_stride=0,
                     reordered=False, curve="3d"):
    """Block routing mask which respects H3's actual video-frame boundaries.

    Sequence-neighbour blocks are merely adjacent spatial strips in H3.  A
    temporal radius therefore has to be expressed in frame coordinates: every
    video query block keeps the complete current/nearby frame(s) exact.  The
    map is tiny (272x272 bytes for the common 17k-token workflow).
    """
    if span is None:
        return None
    video_start, video_stop = bounds
    _start, _stop, grid = span
    frames, height, width = grid
    per_frame = height * width
    if frames * per_frame != video_stop - video_start:
        return None
    key = (video_start, video_stop, grid, int(radius), int(landmark_stride),
           str(device), reordered, curve)
    hit = _ROUTE_CACHE.get(key)
    if hit is not None:
        return hit

    labels = torch.full((video_stop,), -1000000, device=device, dtype=torch.int32)
    video_labels = torch.arange(frames, device=device, dtype=torch.int32).repeat_interleave(per_frame)
    landmark_rows = (video_labels.remainder(int(landmark_stride)) == 0) \
        if landmark_stride else torch.zeros_like(video_labels, dtype=torch.bool)
    if reordered:
        perm, _inverse = _perm_for(grid, curve, device, video_start)
        video_labels = video_labels[perm]
        landmark_rows = landmark_rows[perm]
    labels[video_start:video_stop] = video_labels
    landmarks = torch.zeros((video_stop,), device=device, dtype=torch.bool)
    landmarks[video_start:video_stop] = landmark_rows
    blocks = (video_stop + BLOCK_SIZE - 1) // BLOCK_SIZE
    pad = blocks * BLOCK_SIZE - video_stop
    if pad:
        labels = torch.cat((labels, labels.new_full((pad,), -1000000)))
        landmarks = torch.cat((landmarks, landmarks.new_zeros((pad,))))
    labels = labels.view(blocks, BLOCK_SIZE)
    block_min = torch.where(labels >= 0, labels, labels.new_full((), 1000000)).min(dim=1).values
    block_max = labels.max(dim=1).values
    is_video = block_max >= 0
    landmark_blocks = landmarks.view(blocks, BLOCK_SIZE).any(dim=1)
    # Two frame intervals are within radius when neither is farther than radius
    # beyond the other. Boundary blocks can contain both prefix and video rows;
    # treating them as video is conservative and agrees with the exact sink.
    distance = torch.maximum(
        block_min[:, None] - block_max[None, :],
        block_min[None, :] - block_max[:, None],
    ).clamp_min_(0)
    route = ((distance <= int(radius)) & is_video[:, None] & is_video[None, :]).to(torch.uint8)
    # Global temporal anchors prevent identity/composition drift without making
    # the full long-range video matrix exact. They are KV landmarks only; every
    # video query sees them, while their query rows follow the normal policy.
    route |= (is_video[:, None] & landmark_blocks[None, :]).to(torch.uint8)
    _ROUTE_CACHE[key] = route
    return route


def _perm_for(grid, curve, device, start):
    """Morton permutation for a video span starting at absolute row ``start``.

    The kernels block from absolute position 0, so a span that does not start on
    a block boundary splits every Z-order cell across two blocks, joining
    opposite ends of the volume. Rotating by the misalignment realigns the
    cells; the ragged group it displaces lands in the block shared with the
    conditioning rows, which the exact-KV sink already keeps exact.
    """
    pad = (-int(start)) % BLOCK_SIZE
    key = (tuple(grid), curve, str(device), pad)
    hit = _DEVICE_CACHE.get(key)
    if hit is None:
        perm, inverse = morton_perm(grid, device, curve)
        if pad:
            perm = torch.roll(perm, pad)
            inverse = torch.argsort(perm)
        hit = (perm, inverse)
        _DEVICE_CACHE[key] = hit
    return hit


def _video_span(layout, latent_t, latent_h, latent_w):
    """(start, stop, grid) for the target video segment, or None.

    The grid is stored rather than a permutation so the curve can be changed
    between runs without rebuilding layouts.
    """
    segments = getattr(layout, "segments", None)
    if not segments:
        return None
    span = next(((a, b) for a, b, kind in segments if kind == "video"), None)
    if span is None:
        return None
    start, stop = span
    grid = (int(latent_t), int(latent_h) // 2, int(latent_w) // 2)
    if grid[0] * grid[1] * grid[2] != stop - start:
        logging.info(f"[sol_attn] H3 Morton skipped: video segment {stop - start} rows "
                     f"does not match grid {grid}")
        return None
    return start, stop, grid


def _patch_packed_layout(module):
    """Register the video span of every PackedLayout built, without mutating it."""
    layout_cls = getattr(module, "PackedLayout", None)
    if layout_cls is None:
        raise RuntimeError(f"{module.__name__} has no PackedLayout")
    if id(layout_cls) in _PATCHED_LAYOUTS:
        return
    original_init = layout_cls.__init__

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, *args, **kwargs):
        original_init(self, text_len, latent_t, latent_h, latent_w, audio_t, *args, **kwargs)
        try:
            span = _video_span(self, latent_t, latent_h, latent_w)
        except Exception as exc:                      # never break model construction
            logging.info(f"[sol_attn] H3 Morton span resolution failed: {exc}")
            span = None
        bounds = next(((a, b) for a, b, kind in getattr(self, "segments", []) or []
                       if kind == "video"), None)
        audio_span = next(((a, b) for a, b, kind in getattr(self, "segments", []) or []
                           if kind == "audio"), None)
        if torch.is_tensor(getattr(self, "position_ids", None)) and bounds is not None:
            _SPANS[id(self.position_ids)] = (self, bounds, span, audio_span)

    layout_cls.__init__ = __init__
    _PATCHED_LAYOUTS.add(id(layout_cls))


def install_h3_morton(model):
    """Idempotently install the reorder. Inert without transformer_options['sol_morton']."""
    if id(model) in _INSTALLED:
        return
    for attr in ("rope_freqs", "_forward", "blocks"):
        if not hasattr(model, attr):
            raise RuntimeError(f"MiniMax-H3 Morton needs .{attr} on the diffusion model")

    _patch_packed_layout(sys.modules[type(model).__module__])

    original_forward = model._forward
    original_rope_freqs = model.rope_freqs

    def _forward(x, timestep, context, transformer_options={}, **kwargs):
        previous = getattr(model, "_sol_morton_active", False)
        model._sol_morton_active = bool(transformer_options.get("sol_morton"))
        model._sol_morton_curve = transformer_options.get("sol_morton_curve", "3d")
        model._sol_transformer_options = transformer_options
        try:
            return original_forward(x, timestep, context,
                                    transformer_options=transformer_options, **kwargs)
        finally:
            model._sol_morton_active = previous
            model._sol_morton_span = None
            model._sol_morton_state = None
            model._sol_transformer_options = None
            transformer_options.pop("sol_h3_video_span", None)
            transformer_options.pop("sol_h3_audio_query_spans", None)
            transformer_options.pop("sol_h3_protected_blocks", None)

    def rope_freqs(position_ids, device):
        """Publish the layout only. Permuting happens in the block hook, which is
        the single place that sees the tokens and the rope table together."""
        model._sol_morton_span = None
        model._sol_morton_state = None
        entry = _SPANS.get(id(position_ids))
        if entry is None:
            _log_once("no layout registered; Morton and the conditioning sink are inactive")
            return original_rope_freqs(position_ids, device)

        _layout, bounds, span, audio_span = entry
        # Publish the video-segment boundary so the attention override can keep
        # the conditioning rows (text / audio / reference) exact.
        options = getattr(model, "_sol_transformer_options", None)
        if options is not None:
            options["sol_h3_video_span"] = bounds
            if audio_span is not None:
                options["sol_h3_audio_query_spans"] = (audio_span,)
            if options.get("sol_h3_coordinate_routing"):
                route = _frame_route_map(
                    bounds, span, options.get("sol_h3_temporal_radius", 1), device,
                    landmark_stride=options.get("sol_h3_landmark_stride", 0),
                    reordered=bool(getattr(model, "_sol_morton_active", False)),
                    curve=getattr(model, "_sol_morton_curve", "3d"),
                )
                if route is not None:
                    options["sol_h3_protected_blocks"] = route
                    _log_once(
                        f"coordinate routing {tuple(route.shape)}, protected density "
                        f"{float(route.float().mean()):.3f}"
                    )
        if getattr(model, "_sol_morton_active", False):
            if span is None:
                _log_once("video grid does not match the segment; Morton inactive")
            else:
                model._sol_morton_span = span
        return original_rope_freqs(position_ids, device)

    def pre_hook(module, args):
        # Blocks are called as block(h, t_emb, mod_segments, rope_freqs, ...).
        if len(args) < 4:
            return None

        if module is not first:
            state = getattr(model, "_sol_morton_state", None)
            if state is None:
                return None
            # Hidden states are already reordered; reuse the table built once.
            return args[:3] + (state[1],) + tuple(args[4:])

        model._sol_morton_state = None
        span = getattr(model, "_sol_morton_span", None)
        if span is None:
            return None
        start, stop, grid = span
        h, rope = args[0], args[3]

        # One decision, both tensors in hand. Splitting this across rope_freqs
        # and the hook is what corrupts output when the two guards disagree.
        if not torch.is_tensor(h) or h.ndim != 2 or h.shape[0] < stop:
            _log_once(f"hidden states do not cover the video span {(start, stop)}; inactive")
            return None
        if not torch.is_tensor(rope) or rope.ndim < 2 or rope.shape[1] != h.shape[0]:
            _log_once(f"rope table {tuple(rope.shape) if torch.is_tensor(rope) else type(rope)} "
                      f"does not match {h.shape[0]} tokens; inactive")
            return None

        curve = getattr(model, "_sol_morton_curve", "3d")
        perm, inverse = _perm_for(grid, curve, h.device, start)
        h = h.clone()
        h[start:stop] = h[start:stop][perm]

        # rope_rotation_table is elementwise per row, so permuting the table is
        # identical to permuting position_ids -- and keeps the decision here.
        full = torch.arange(rope.shape[1], device=rope.device)
        full[start:stop] = perm + start
        rope = rope.index_select(1, full)

        model._sol_morton_state = (inverse, rope)
        return (h,) + args[1:3] + (rope,) + tuple(args[4:])

    def post_hook(_module, _args, output):
        state = getattr(model, "_sol_morton_state", None)
        span = getattr(model, "_sol_morton_span", None)
        model._sol_morton_state = None
        if state is None or span is None:
            return None
        start, stop, _grid = span
        inverse, _rope = state
        if not torch.is_tensor(output) or output.shape[0] < stop:
            logging.error("[sol_attn] H3 Morton: block stack changed the token count "
                          f"({tuple(output.shape)}); cannot restore order")
            return None
        output = output.clone()
        output[start:stop] = output[start:stop][inverse]
        return output

    first = model.blocks[0]
    model._forward = _forward
    model.rope_freqs = rope_freqs
    for block in model.blocks:
        block.register_forward_pre_hook(pre_hook)
    model.blocks[-1].register_forward_hook(post_hook)
    _INSTALLED.add(id(model))


_logged = set()


def _log_once(message):
    if message not in _logged:
        _logged.add(message)
        logging.info(f"[sol_attn] H3 Morton: {message}")


install_h3_hooks = install_h3_morton

__all__ = ["install_h3_morton", "install_h3_hooks"]

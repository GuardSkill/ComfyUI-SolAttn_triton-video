"""Sol-Attn (arXiv 2607.24027) as a ComfyUI attention override.

The "Patch Sol-Attn" node installs an ``optimized_attention_override`` on the
model, keeping the patch per-model and giving it the sigma schedule for dense
warm-up steps. ``SOL_ATTN=1`` installs a global override for CLI benchmarks.
"""

import logging
import os
import os
import re
import json
from functools import partial

import torch
import torch.nn.functional as F

from comfy.ldm.modules.attention import (
    attention_pytorch,
    register_attention_function,
    wrap_attn,
)
from comfy.patcher_extension import CallbacksMP
from comfy_api.latest import ComfyExtension, io

from ._autotune_log import set_verbose as _set_autotune_verbose

try:
    from ._tri_fwd import sol_attn as _sol_attn_kernel, _has_tma
    _IMPORT_ERROR = None
except Exception as exc:  # triton / torch version issues
    _sol_attn_kernel = None
    _has_tma = None
    _IMPORT_ERROR = exc

try:
    from ._int8_fwd import sol_attn_int8 as _sol_attn_int8_kernel
    _INT8_IMPORT_ERROR = None
except Exception as exc:
    _sol_attn_int8_kernel = None
    _INT8_IMPORT_ERROR = exc

try:
    from ._int8_fwd_legacy import sol_attn_int8 as _sol_attn_int8_legacy_kernel
except Exception:
    _sol_attn_int8_legacy_kernel = None


HEAD_DIM = 128

_stats = {"sparse": 0, "dense_fallback": 0, "outside_range": 0,
          "dense_block": 0, "errors": 0}
_seen = set()
_BLOCK_INDEX_HOOKED = set()


def parse_blocks(spec, count):
    """Parse "0-3,47,-1" into absolute block indices; negatives count from the end."""
    out = set()
    for part in "".join(str(spec).split()).split(","):   # tolerate any whitespace
        if not part:
            continue
        match = re.fullmatch(r"(-?\d+)(?:-(-?\d+))?", part)
        if match is None:
            raise ValueError(f"cannot parse block spec {part!r}; "
                             "use indices and ranges like '0-3,47,-1'")
        first = int(match.group(1))
        last = first if match.group(2) is None else int(match.group(2))
        first = first if first >= 0 else count + first
        last = last if last >= 0 else count + last
        if first > last:
            first, last = last, first
        out.update(range(max(first, 0), min(last, count - 1) + 1))
    return frozenset(out)


def parse_tau_profile(spec, count):
    """Parse "0-30=2.0; 39-42=0.9" into {block: tau}.

    Entries are separated by ';' or newlines, so a multiline text node works as
    well as a single line, and '#' starts a comment. Blocks not listed keep the
    node's base tau; the block side takes dense_blocks syntax, so "0-2,47=1.8"
    is valid. Later entries win where they overlap.
    """
    profile = {}
    for entry in re.split(r"[;\n]", str(spec)):
        entry = entry.split("#", 1)[0].strip()
        if not entry:
            continue
        blocks, sep, value = entry.partition("=")
        if not sep:
            raise ValueError(f"tau_profile entry {entry!r} needs '=', e.g. '39-42=0.9'")
        try:
            level = float(value)
        except ValueError:
            raise ValueError(f"tau_profile entry {entry!r} has a non-numeric tau")
        for block in parse_blocks(blocks, count):
            profile[block] = level
    return profile


def _install_block_index(model):
    """Publish the running block index into transformer_options."""
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return False
    if id(model) in _BLOCK_INDEX_HOOKED:
        return True

    def make_hook(index):
        def hook(_module, _args, kwargs):
            options = kwargs.get("transformer_options")
            if isinstance(options, dict):
                options["sol_block"] = index
            return None
        return hook

    for index, block in enumerate(blocks):
        block.register_forward_pre_hook(make_hook(index), with_kwargs=True)
    _BLOCK_INDEX_HOOKED.add(id(model))
    return True


_probe = {}
_probe_timings = {}


def _probe_record(block, sparse_out, reference):
    """Accumulate this block's sparse-vs-dense relative error."""
    ref = reference.float()
    norm = ref.norm()
    if norm > 0:
        entry = _probe.setdefault(block, [0.0, 0])
        entry[0] += float((sparse_out.float() - ref).norm() / norm)
        entry[1] += 1


def log_probe_summary(*_args, output_path=None, **_kwargs):
    """Per-block sensitivity, worst first. Blocks at the top are the ones worth
    listing in dense_blocks."""
    if not _probe:
        return
    rows = sorted(((total / count, block, count)
                   for block, (total, count) in _probe.items()), reverse=True)
    logging.info("[sol_attn] block sensitivity to sparsification (worst first); "
                 "put the top entries in dense_blocks:")
    report = []
    torch.cuda.synchronize()
    for error, block, count in rows:
        timings = _probe_timings.get(block, [])
        sparse_ms = [start.elapsed_time(stop) for start, stop, _, _ in timings]
        reference_ms = [start.elapsed_time(stop) for _, _, start, stop in timings]
        logging.info(f"[sol_attn]   block {block:3d}  rel err {error:.4f}  "
                     f"({count} calls)")
        report.append({
            "block": block,
            "relative_l2": error,
            "calls": count,
            "sol_ms": sum(sparse_ms) / len(sparse_ms) if sparse_ms else None,
            "sage_ms": sum(reference_ms) / len(reference_ms) if reference_ms else None,
        })
    if output_path:
        from pathlib import Path
        destination = Path(output_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({
            "gpu": torch.cuda.get_device_name(),
            "per_block": sorted(report, key=lambda row: row["block"]),
        }, indent=2) + "\n")
        logging.info(f"[sol_attn] wrote Sage-vs-Sol block probe to {destination}")
    _probe.clear()
    _probe_timings.clear()


def sol_attn_stats():
    """Dispatch counters since process start (or last reset)."""
    return dict(_stats)


def reset_sol_attn_stats():
    for key in _stats:
        _stats[key] = 0
    _seen.clear()


def _log_once(key, message):
    if key not in _seen:
        _seen.add(key)
        logging.info(f"[sol_attn] {message}")


def _log_kernel_failure(exc):
    # Full traceback on the first occurrence of each distinct failure; a short
    # line on repeats so a failing run stays diagnosable without spam.
    key = ("kernel_failure", type(exc).__name__, str(exc))
    first = key not in _seen
    _seen.add(key)
    logging.error(f"[sol_attn] kernel failed ({exc}); falling back", exc_info=first)


def _ineligible(q, k, mask, dim_head, min_tokens):
    """Why this call can't use Sol-Attn, or None if it can. q/k are BTHD."""
    if _sol_attn_kernel is None:
        return f"kernel import failed: {_IMPORT_ERROR}"
    if q.device.type != "cuda":
        return "not cuda"
    if q.dtype not in (torch.float16, torch.bfloat16):
        return f"dtype {q.dtype} (kernel requires fp16 or bf16)"
    if dim_head != HEAD_DIM:
        return f"head_dim {dim_head} != 128"
    if mask is not None:
        return "masked attention"
    if q.shape[1] != k.shape[1]:
        return "cross-attention (kept dense)"
    if q.shape != k.shape:
        # GQA or any other q/k mismatch would silently index wrong.
        return f"q/k shape mismatch {tuple(q.shape)} vs {tuple(k.shape)}"
    if q.shape[1] < min_tokens:
        return f"seq {q.shape[1]} < {min_tokens}"
    return None


def _run(q, k, v, heads, skip_reshape, skip_output_reshape, scale,
         tau, min_tokens, verbose, int8_qk=False, sink_blocks=(0, 0),
         sink_q=(0, 0), use_tma=False, int8_pv=True, local_blocks=1,
         int8_scope="distant", protected_blocks=None):
    """Returns the attention output, or None if this call should stay dense."""
    if skip_reshape:
        b, _, _, dim_head = q.shape          # BHND
        qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
    else:
        b, _, dim_head = q.shape             # B, N, heads*dim_head
        dim_head //= heads
        qs, ks, vs = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

    reason = _ineligible(qs, ks, None, dim_head, min_tokens)
    if reason is not None:
        _stats["dense_fallback"] += 1
        if verbose:
            _log_once((tuple(qs.shape), reason), f"dense {tuple(qs.shape)}: {reason}")
        return None

    # No contiguous() here: the kernels take strides, so H3's interleaved qkv
    # views go in without copies.
    legacy_fast = (
        int8_qk and int8_scope == "all" and local_blocks == 1
        and protected_blocks is None and _sol_attn_int8_legacy_kernel is not None
    )
    if legacy_fast:
        # The original pointer kernel is substantially faster on SM89. Audio
        # query rows are still repaired exactly by the wrapper below.
        out = _sol_attn_int8_legacy_kernel(
            qs, ks, vs,
            scale=scale, tau=tau, sink_blocks=sink_blocks, sink_q=sink_q,
            use_tma=use_tma, int8_pv=int8_pv,
        )
    else:
        extra = {
            "int8_pv": int8_pv,
            "mixed_exact": int8_scope == "distant",
        } if int8_qk else {}
        kernel = _sol_attn_int8_kernel if int8_qk else _sol_attn_kernel
        out = kernel(
            qs, ks, vs,
            scale=scale, tau=tau, sink_blocks=sink_blocks, sink_q=sink_q,
            local_blocks=local_blocks, use_tma=use_tma, **extra,
            protected_blocks=protected_blocks,
        )  # BTHD
    _stats["sparse"] += 1
    if verbose:
        mode = "int8" if int8_qk else "bf16"
        # The kernels also require SM90+ and a Triton with TensorDescriptor, so
        # report the path actually taken rather than what was asked for.
        path = "tma" if (use_tma and _has_tma(qs.device)) else "pointer"
        _log_once((tuple(qs.shape), "sparse", mode, path),
                  f"sparse {tuple(qs.shape)} tau={tau} {mode} {path}")

    if skip_output_reshape:
        result = out.transpose(1, 2)         # BHND
    else:
        result = out.reshape(b, -1, heads * dim_head)
    return result


def _run_sage_dense(q, k, v, heads, skip_reshape, skip_output_reshape, scale,
                    min_tokens, verbose):
    """Full-coverage Sage attention using its architecture-native fast path."""
    try:
        from sageattention import sageattn
    except Exception as exc:
        raise RuntimeError(f"SageAttention CUDA backend unavailable: {exc}") from exc
    if skip_reshape:
        batch, _, _, dim_head = q.shape
        qs, ks, vs = (tensor.transpose(1, 2) for tensor in (q, k, v))
    else:
        batch, _, width = q.shape
        dim_head = width // heads
        qs, ks, vs = (tensor.view(batch, -1, heads, dim_head) for tensor in (q, k, v))
    reason = _ineligible(qs, ks, None, dim_head, min_tokens)
    if reason is not None:
        _stats["dense_fallback"] += 1
        return None
    # On RTX 4090 (SM89) Sage auto dispatches to QK-INT8/PV-FP8 with
    # fp32+fp16 accumulation, matching KJNodes' proven `auto` workflow. The
    # selected audio query rows are repaired in BF16 below.
    out = sageattn(
        qs, ks, vs, tensor_layout="NHD", is_causal=False,
        sm_scale=scale,
    ).to(v.dtype)
    _stats["sparse"] += 1
    if verbose:
        _log_once((tuple(qs.shape), "sage_dense"),
                  f"Sage dense full-KV {tuple(qs.shape)} architecture-auto")
    if skip_output_reshape:
        return out.transpose(1, 2)
    return out.reshape(batch, -1, heads * dim_head)


def _overwrite_exact_queries(out, q, k, v, heads, spans, *,
                             skip_reshape, skip_output_reshape, scale):
    """Replace selected H3 query rows with exact BF16 rectangular attention."""
    if not spans:
        return out
    if skip_reshape:
        qh, kh, vh = q, k, v  # BHND
    else:
        batch, tokens, width = q.shape
        dim = width // heads
        qh, kh, vh = (
            tensor.view(batch, tokens, heads, dim).transpose(1, 2)
            for tensor in (q, k, v)
        )
    for start, stop in spans:
        if not (0 <= start < stop <= qh.shape[2]):
            continue
        exact = F.scaled_dot_product_attention(
            qh[:, :, start:stop], kh, vh, scale=scale
        )
        if skip_output_reshape:
            out[:, :, start:stop] = exact
        else:
            out[:, start:stop] = exact.transpose(1, 2).flatten(2, 3)
    return out


BLOCK_SIZE = int(os.environ.get("SOL_BLOCK_SIZE", "64"))


def _sink_blocks(transformer_options, tokens, mode):
    """(exact-KV blocks, dense-query blocks) for MiniMax-H3's conditioning rows.

    H3 packs [text][cond][ref][audio][video] into one sequence; sparsifying the
    conditioning rows costs sync and prompt adherence. exact_kv measures ~3%,
    exact_kv_and_rows ~17%, so exact-KV is the default and rows are opt-in.
    """
    if mode == "off":
        return (0, 0), (0, 0)
    span = (transformer_options or {}).get("sol_h3_video_span")
    if span is None:
        return (0, 0), (0, 0)
    video_start, video_stop = span
    if tokens < video_stop or video_start <= 0:
        return (0, 0), (0, 0)
    blocks = (0, (video_start + BLOCK_SIZE - 1) // BLOCK_SIZE)
    return blocks, (blocks if mode == "exact_kv_and_rows" else (0, 0))


def make_override(tau=1.0, min_tokens=4096,
                  sigma_start=None, sigma_end=None, verbose=False,
                  int8_qk=False, sink_conditioning="exact_kv", use_tma=False,
                  dense_blocks=frozenset(), tau_profile=None, int8_pv=True,
                  local_blocks=1, exact_audio_queries=True,
                  int8_scope="distant",
                  int8_dense_blocks=frozenset(),
                  int8_start=None, int8_end=None,
                  coordinate_routing=True, temporal_radius=1, landmark_stride=4,
                  backend="sol_sparse", sol_blocks=frozenset(), sage_blocks=frozenset(),
                  previous=None):
    """Build an optimized_attention_override callable.

    ``previous`` chains any override already installed on the model: every path
    that declines hands off to it first, falling through to ``func`` only if
    there is none.
    """

    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):

        def dense():
            target = func if previous is None else partial(previous, func)
            return target(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                          skip_reshape=skip_reshape,
                          skip_output_reshape=skip_output_reshape, **kwargs)

        if mask is not None:
            _stats["dense_fallback"] += 1
            return dense()

        # Depth gates, the counterpart of the sigma window below. Sensitivity to
        # sparsification varies several-fold across depth, so a block can be kept
        # dense outright or given its own tau.
        block = None
        if dense_blocks or int8_dense_blocks or tau_profile or backend == "tri_policy":
            block = kwargs.get("transformer_options", {}).get("sol_block")
        if block in dense_blocks:
            _stats["dense_block"] += 1
            return dense()
        block_tau = tau_profile.get(block, tau) if tau_profile else tau
        effective_backend = backend
        if backend == "tri_policy":
            if block in sol_blocks:
                effective_backend = "sol_sparse"
            elif block in sage_blocks:
                effective_backend = "sage_dense"
            else:
                _stats["dense_block"] += 1
                return dense()

        # Keep the dense warm-up/end window for every approximate attention
        # backend.  On RTX 4090 Sage is slower than the native dense path at
        # H3's shape, so enabling it outside this window is a regression.
        if sigma_start is not None or sigma_end is not None:
            sigmas = kwargs.get("transformer_options", {}).get("sigmas")
            if sigmas is not None:
                sigma = float(sigmas[0])
                if (sigma_start is not None and sigma > sigma_start) or \
                   (sigma_end is not None and sigma < sigma_end):
                    _stats["outside_range"] += 1
                    return dense()

        # INT8 is the main speed path, but a small number of video-model
        # layers and denoise endpoints can amplify a one-frame routing error
        # into hand/face ghosts.  Keep Sol routing enabled there while making
        # only its QK/PV arithmetic BF16.  This is deliberately independent
        # of dense_blocks: it retains most of the sparse-attention speed.
        use_int8 = int8_qk and block not in int8_dense_blocks
        sigmas = kwargs.get("transformer_options", {}).get("sigmas")
        if use_int8 and sigmas is not None and (int8_start is not None or int8_end is not None):
            sigma = float(sigmas[0])
            if (int8_start is not None and sigma > int8_start) or \
               (int8_end is not None and sigma < int8_end):
                use_int8 = False
                _stats["int8_safety_fallback"] = _stats.get("int8_safety_fallback", 0) + 1

        tokens = q.shape[2] if skip_reshape else q.shape[1]
        sink, sink_q = _sink_blocks(kwargs.get("transformer_options"), tokens,
                                    sink_conditioning)
        protected = kwargs.get("transformer_options", {}).get("sol_h3_protected_blocks") \
            if coordinate_routing else None
        if verbose and sink != (0, 0):
            _log_once((tokens, sink, sink_q),
                      f"conditioning sink: KV blocks {sink} exact, dense query blocks {sink_q}")

        try:
            if effective_backend == "sage_dense":
                out = _run_sage_dense(
                    q, k, v, heads, skip_reshape, skip_output_reshape,
                    kwargs.get("scale", None), min_tokens, verbose,
                )
            else:
                out = _run(q, k, v, heads, skip_reshape, skip_output_reshape,
                           kwargs.get("scale", None), block_tau,
                           min_tokens, verbose, use_int8, sink, sink_q, use_tma,
                           int8_pv, local_blocks, int8_scope, protected)
        except Exception as exc:
            _stats["errors"] += 1
            _log_kernel_failure(exc)
            return dense()
        if out is None:
            return dense()
        if exact_audio_queries:
            spans = kwargs.get("transformer_options", {}).get(
                "sol_h3_audio_query_spans", ()
            )
            out = _overwrite_exact_queries(
                out, q, k, v, heads, spans,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                scale=kwargs.get("scale", None),
            )
        return out

    override._sol_previous = previous
    return override


def make_probe_override(inner):
    """Wrap an installed override so every call is computed both ways.

    Returns the dense result, so each block is measured against a clean input;
    returning the sparse one would let early error compound and inflate every
    later block's number.
    """

    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        common = dict(mask=mask, attn_precision=attn_precision,
                      skip_reshape=skip_reshape,
                      skip_output_reshape=skip_output_reshape, **kwargs)
        sparse_start, sparse_stop = torch.cuda.Event(True), torch.cuda.Event(True)
        reference_start, reference_stop = torch.cuda.Event(True), torch.cuda.Event(True)
        sparse_start.record()
        sparse = inner(func, q, k, v, heads, **common)
        sparse_stop.record()
        reference_start.record()
        previous = getattr(inner, "_sol_previous", None)
        reference = (func(q, k, v, heads, **common) if previous is None else
                     previous(func, q, k, v, heads, **common))
        reference_stop.record()
        block = kwargs.get("transformer_options", {}).get("sol_block")
        _probe_timings.setdefault(block, []).append(
            (sparse_start, sparse_stop, reference_start, reference_stop)
        )
        _probe_record(block,
                      sparse, reference)
        return reference

    return override


def _compose_module_patch(module, patched_forward):
    """Gate an object-patched attention forward (e.g. KJNodes' mem-efficient
    Sage): calls Sol-Attn would take run the stock forward and reach the
    override; the rest keeps the patch. Gate params come from
    transformer_options["sol_compose"]; when absent the patch runs as-is.
    """
    stock = type(module).forward

    def forward(*args, **kwargs):
        options = kwargs.get("transformer_options")
        if not isinstance(options, dict):
            options = next((a for a in args if isinstance(a, dict) and "sol_compose" in a), {})
        gate = options.get("sol_compose")
        x = args[0] if args else None
        # KJNodes' low-VRAM block patch hands x over in a single-item list.
        tensor = x[0] if isinstance(x, list) and len(x) == 1 and torch.is_tensor(x[0]) else x
        take = gate is not None and torch.is_tensor(tensor) and tensor.device.type == "cuda" \
            and tensor.dtype == torch.bfloat16 and tensor.ndim in (2, 3)
        if take:
            # H3 packs tokens first (s, dim); Wan/LTX2 are batch-first.
            tokens = tensor.shape[0] if tensor.ndim == 2 else tensor.shape[1]
            take = tokens >= gate["min_tokens"]
        if take:
            sigmas = options.get("sigmas")
            if sigmas is not None:
                sigma = float(sigmas[0])
                take = not (sigma > gate["sigma_start"] or sigma < gate["sigma_end"])
        if take:
            delegate = options.get("sol_take_forward")
            if delegate is not None:
                # a cooperating patch's forward that reaches optimized_attention while
                # keeping its own low-VRAM behavior; preferred over the stock forward
                return delegate(module, *args, **kwargs)
            if tensor is not x:
                x.clear()  # the stock forward wants the tensor; consume the hand-off list
                args = (tensor,) + args[1:]
            return stock(module, *args, **kwargs)
        return patched_forward(*args, **kwargs)

    forward._sol_composed = True
    return forward


_COMPOSE_HOOKED = set()


def _install_compose_hooks(model, attn_attr):
    """Compose at sampling time, once all object patches are applied: a node
    downstream of ours overwrites the same object-patch key, so execute-time
    composition alone loses. The pre-hooks re-wrap any foreign attn forward
    before each block runs; inert unless sol_compose is published.
    """
    if id(model) in _COMPOSE_HOOKED:
        return

    def pre_hook(block, args):
        attn = getattr(block, attn_attr, None)
        if attn is None:
            return None
        fwd = attn.__dict__.get("forward")
        if fwd is None or getattr(fwd, "_sol_composed", False):
            return None
        if getattr(fwd, "_uses_optimized_attention", False):
            return None  # patch routes through optimized_attention; the override composes directly
        if getattr(fwd, "__func__", None) is type(attn).forward:
            return None  # unpatch leaves the stock forward as an instance attr
        attn.forward = _compose_module_patch(attn, fwd)
        _log_once(("composed", attn_attr),
                  f"composing with a patched {attn_attr}.forward; Sol-Attn takes "
                  "eligible self-attention calls, the patch keeps the rest")
        return None

    for block in model.blocks:
        block.register_forward_pre_hook(pre_hook)
    _COMPOSE_HOOKED.add(id(model))


class SolAttnVideoPatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SolAttnVideoPatch",
            display_name="Patch Sol-Attn Video",
            is_experimental=True,
            category="sol_attn",
            description="Sparsify self-attention with Sol-Attn (arXiv 2607.24027), with "
                        "optional Morton token reordering, INT8 QK, and an exact-KV sink "
                        "for MiniMax-H3's packed conditioning rows. bf16 + head_dim 128 "
                        "only; everything else falls back to the existing attention backend.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("backend", options=["sol_sparse", "sage_dense", "tri_policy"],
                               default="sol_sparse",
                               tooltip="sol_sparse uses block summaries and is faster but failed "
                                       "the H3 visual gate at useful depth. sage_dense keeps every "
                                       "KV block and only quantizes QK to INT8; safer for H3."),
                io.Float.Input("tau", default=1.3, min=0.0, max=4.0, step=0.05,
                               tooltip="Threshold beta. Higher is sparser: 1.0 ~ 16% of "
                                       "blocks kept exact, 1.5 ~ 7%, 2.0 ~ 2.7%."),
                io.Float.Input("start_percent", default=0.2, min=0.0, max=1.0, step=0.01,
                               tooltip="Run dense before this point. The paper uses 0.2."),
                io.Float.Input("end_percent", default=0.9, min=0.0, max=1.0, step=0.01),
                io.Int.Input("min_tokens", default=4096, min=0, max=1 << 20, step=512,
                             tooltip="Sequences shorter than this stay dense."),
                io.Boolean.Input("int8_qk", default=True,
                                 tooltip="INT8 QK in the exact branch (Sage-style: smoothed K, "
                                         "per-token scales). Measured free in quality; helps at "
                                         "tau<=1.5, a net loss at tau>=2.0 where the quantize "
                                         "pass outweighs the shrinking exact branch."),
                io.Combo.Input("sink_conditioning", options=["exact_kv", "exact_kv_and_rows", "off"],
                               default="exact_kv_and_rows",
                               tooltip="MiniMax-H3 only. exact_kv: every query sees the packed "
                                       "text/audio/reference rows exactly (~3% cost). "
                                       "exact_kv_and_rows: also runs those query rows dense, making "
                                       "the generated audio stream exact (~20% cost). "
                                       "No effect on other models."),
                io.Boolean.Input("morton", default=False,
                                 tooltip="Reorder video tokens into Morton (Z-order) so each "
                                         "64-token block is a compact 3D neighbourhood instead of a "
                                         "2-row strip, which makes routing far more accurate at a "
                                         "given density. Exactly neutral for dense attention. "
                                         "Wan and MiniMax-H3 only; logged and skipped elsewhere."),
                io.Combo.Input("morton_curve", options=["3d", "2d_frame"], default="2d_frame",
                               tooltip="3d interleaves t/h/w equally. 2d_frame Z-orders within "
                                       "each frame and leaves frame order alone -- use it when the "
                                       "temporal axis is not uniformly spaced (MiniMax-H3's frame "
                                       "spacing is non-uniform; try this if 3d degrades at some "
                                       "frame counts)."),
                io.Boolean.Input("int8_pv", default=True,
                                 tooltip="Also run the exact branch's P@V in INT8, with a "
                                         "per-row P scale and per-channel V scale. PV and QK "
                                         "cost the same, so this is the other half of the "
                                         "int8 win. Only applies when int8_qk is on."),
                io.Int.Input("local_blocks", default=1, min=0, max=8, step=1,
                             tooltip="Exact block radius around every query block. With H3 "
                                     "3D Morton, radius 1 preserves a compact local "
                                     "spatiotemporal neighborhood; larger radii trade speed "
                                     "for local and temporal fidelity."),
                io.Boolean.Input("exact_audio_queries", default=True,
                                 tooltip="Recompute only H3 target-audio query rows with exact "
                                         "BF16 rectangular attention. This is cheaper than "
                                         "making all conditioning query rows dense."),
                io.Boolean.Input("verbose", default=False),
                io.Boolean.Input("use_tma", default=False,
                                 tooltip="Use the TMA descriptor kernels instead of the "
                                         "pointer ones. Descriptors address strided inputs "
                                         "directly, so this no longer copies q/k/v and peak "
                                         "VRAM matches the pointer path. Off by default "
                                         "because it has not measured faster on any tested "
                                         "GPU. Requires SM90+ and Triton 3.3+; ignored "
                                         "otherwise. 'verbose' logs the path used."),
                io.String.Input("tau_profile", optional=True, force_input=True,
                                tooltip="Per-block tau, overriding the base value. "
                                        "'blocks=tau' entries separated by ';' or newlines, "
                                        "so a multiline text node works: '0-30=2.0' then "
                                        "'39-42=0.9'. '#' starts a comment. Block "
                                        "sensitivity varies several-fold across depth, so "
                                        "one tau either over-serves the insensitive blocks "
                                        "or under-serves the fragile ones — use the Block "
                                        "Probe to find them. Leave unconnected for a single "
                                        "tau everywhere."),
                io.String.Input("dense_blocks", default="",
                                                tooltip="Transformer blocks to keep dense, e.g. '0-2,-1' "
                                                        "for the first three and the last. Negative "
                                                        "indices count from the end. The first and last "
                                                        "blocks are the most approximation-sensitive: "
                                                        "their error reaches the output with no later "
                                                        "block to absorb it. Empty means sparsify all."),
                io.Combo.Input("int8_scope", options=["distant", "all"], default="distant",
                               tooltip="distant keeps conditioning/keyframes and the local "
                                       "temporal radius in BF16, using INT8 only for routed "
                                       "long-range blocks. On RTX 4090 this retains 97% of the "
                                       "all-INT8 speed and is the safer H3 default."),
                io.String.Input("int8_dense_blocks", default="",
                                tooltip="Video safety: use BF16 QK/PV only in these transformer "
                                        "blocks while retaining Sol sparse routing, e.g. '0-3,36-39'. "
                                        "This targets hand/face ghosting with much less cost than "
                                        "putting the blocks in dense_blocks."),
                io.Float.Input("int8_start_percent", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Video safety: keep INT8 off before this denoise percentage; "
                                       "Sol routing remains enabled. 0 means no initial BF16 window."),
                io.Float.Input("int8_end_percent", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Video safety: keep INT8 off after this denoise percentage; "
                                       "1 means no final BF16 window."),
                io.Boolean.Input("coordinate_routing", default=True,
                                 tooltip="MiniMax-H3: protect complete current/neighbor video "
                                         "frames using the true packed (t,h,w) layout. Sequence "
                                         "local_blocks are spatial strips, not temporal neighbors."),
                io.Int.Input("temporal_radius", default=1, min=0, max=32, step=1,
                             tooltip="MiniMax-H3 exact frame radius used by coordinate routing."),
                io.Int.Input("landmark_stride", default=4, min=0, max=32, step=1,
                             tooltip="MiniMax-H3: every Nth latent frame is exact KV for all "
                                       "video queries. 4 preserves global identity cheaply; 0 disables."),
                io.String.Input("sol_blocks", default="12-18,20-22",
                                tooltip="tri_policy only: blocks handled by sparse Sol-Attn."),
                io.String.Input("sage_blocks", default="3-11,19,23-38,40-45",
                                tooltip="tri_policy only: blocks handled by full-KV Sage QK-INT8. "
                                        "All remaining blocks use exact dense BF16 attention."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, tau, start_percent, end_percent,
                min_tokens, int8_qk, sink_conditioning, morton,
                morton_curve, dense_blocks, verbose,
                tau_profile=None, use_tma=False, int8_pv=True,
                local_blocks=1, exact_audio_queries=True,
                int8_scope="distant", coordinate_routing=True,
                int8_dense_blocks="", int8_start_percent=0.0, int8_end_percent=1.0,
                temporal_radius=1, landmark_stride=4,
                backend="sol_sparse", sol_blocks="12-18,20-22",
                sage_blocks="3-11,19,23-38,40-45") -> io.NodeOutput:
        if _sol_attn_kernel is None:
            raise RuntimeError(f"Sol-Attn kernel unavailable: {_IMPORT_ERROR}")
        if int8_qk and _sol_attn_int8_kernel is None:
            raise RuntimeError(f"Sol-Attn INT8 kernel unavailable: {_INT8_IMPORT_ERROR}")

        diffusion_model = model.get_model_object("diffusion_model")
        is_h3 = hasattr(diffusion_model, "rope_freqs") and hasattr(diffusion_model, "_forward")
        is_wan = hasattr(diffusion_model, "rope_encode") and hasattr(diffusion_model, "blocks")

        # H3 publishes its segment layout from the same hooks Morton uses, so the
        # conditioning sink needs them installed even when reordering is off.
        reorder = False
        if is_h3 and (morton or sink_conditioning != "off" or exact_audio_queries or coordinate_routing):
            from ._morton_h3 import install_h3_hooks

            install_h3_hooks(diffusion_model)
            reorder = morton
        elif is_wan and morton:
            from ._morton import install_wan_morton

            install_wan_morton(diffusion_model)
            reorder = True
        elif morton:
            logging.warning(
                f"[sol_attn] Morton reordering skipped: {type(diffusion_model).__name__} "
                "is neither Wan-style nor MiniMax-H3. Sol-Attn itself still applies."
            )

        blocks = getattr(diffusion_model, "blocks", None)
        count = len(blocks) if blocks is not None else 0
        dense = parse_blocks(dense_blocks, count)
        int8_safe = parse_blocks(int8_dense_blocks, count)
        sol_selected = parse_blocks(sol_blocks, count) if backend == "tri_policy" else frozenset()
        sage_selected = parse_blocks(sage_blocks, count) if backend == "tri_policy" else frozenset()
        overlap = sol_selected & sage_selected
        if overlap:
            raise ValueError(f"tri_policy sol_blocks and sage_blocks overlap: {sorted(overlap)}")
        profile = parse_tau_profile(tau_profile or "", count)
        if (dense or int8_safe or profile or backend == "tri_policy") and not _install_block_index(diffusion_model):
            logging.warning(
                f"[sol_attn] per-block Sol policy ignored: "
                f"{type(diffusion_model).__name__} has no .blocks list to index")
            dense, profile = frozenset(), {}
            sol_selected, sage_selected = frozenset(), frozenset()
        if dense:
            logging.info(f"[sol_attn] keeping blocks {sorted(dense)} dense of {count}")
        if int8_safe:
            logging.info(f"[sol_attn] keeping INT8 off for blocks {sorted(int8_safe)} of {count}")
        if profile:
            levels = {}
            for block, level in sorted(profile.items()):
                levels.setdefault(level, []).append(block)
            logging.info("[sol_attn] per-block tau: "
                         + ", ".join(f"{lv} on {len(bs)} block(s)"
                                     for lv, bs in sorted(levels.items()))
                         + f"; base {tau} elsewhere")

        model_sampling = model.get_model_object("model_sampling")
        sigma_start = float(model_sampling.percent_to_sigma(start_percent))
        sigma_end = float(model_sampling.percent_to_sigma(end_percent))

        m = model.clone()
        previous = m.model_options["transformer_options"].get("optimized_attention_override")
        if previous is not None:
            logging.info("[sol_attn] chaining onto an existing attention override; "
                         "Sol-Attn takes first refusal and delegates everything else to it")

        # Forward-level patches bypass optimized_attention entirely, so there is
        # nothing to chain onto -- gate each one instead (see _compose_module_patch).
        composed = []
        for key, patched in list(m.object_patches.items()):
            if not key.endswith(".forward"):
                continue
            owner = key.rsplit(".", 2)[-2].lower()
            if "attn" not in owner or "cross" in owner or owner == "attn2":
                continue  # Sol-Attn never takes cross-attention; leave it patched
            if getattr(patched, "_uses_optimized_attention", False):
                continue  # patch routes through optimized_attention; the override composes directly
            module = m.get_model_object(key[: -len(".forward")])
            m.add_object_patch(key, _compose_module_patch(module, patched))
            composed.append(key)
        if composed:
            logging.info(
                f"[sol_attn] composed with {len(composed)} object-patched attention "
                f"forward(s) (e.g. {composed[0]}): Sol-Attn takes eligible "
                "self-attention calls, the existing patch keeps the rest")
        # Downstream nodes overwrite the same keys; the hooks catch those at sampling time.
        if is_h3:
            _install_compose_hooks(diffusion_model, "attn")
        elif is_wan:
            _install_compose_hooks(diffusion_model, "self_attn")

        m.model_options["transformer_options"]["sol_compose"] = {
            "sigma_start": sigma_start, "sigma_end": sigma_end,
            "min_tokens": min_tokens}
        m.model_options["transformer_options"]["optimized_attention_override"] = \
            make_override(tau=tau, min_tokens=min_tokens,
                          sigma_start=sigma_start, sigma_end=sigma_end,
                          verbose=verbose, int8_qk=int8_qk,
                          sink_conditioning=sink_conditioning,
                          use_tma=use_tma, dense_blocks=dense,
                          tau_profile=profile, int8_pv=int8_pv,
                          local_blocks=local_blocks,
                          exact_audio_queries=exact_audio_queries,
                          int8_scope=int8_scope,
                          int8_dense_blocks=int8_safe,
                          int8_start=float(model_sampling.percent_to_sigma(int8_start_percent))
                          if int8_start_percent > 0.0 else None,
                          int8_end=float(model_sampling.percent_to_sigma(int8_end_percent))
                          if int8_end_percent < 1.0 else None,
                          coordinate_routing=coordinate_routing,
                          temporal_radius=temporal_radius,
                          landmark_stride=landmark_stride,
                          backend=backend, sol_blocks=sol_selected,
                          sage_blocks=sage_selected,
                          previous=previous)
        if is_h3 and coordinate_routing:
            m.model_options["transformer_options"]["sol_h3_coordinate_routing"] = True
            m.model_options["transformer_options"]["sol_h3_temporal_radius"] = temporal_radius
            m.model_options["transformer_options"]["sol_h3_landmark_stride"] = landmark_stride
        if reorder:
            m.model_options["transformer_options"]["sol_morton"] = True
            m.model_options["transformer_options"]["sol_morton_curve"] = morton_curve
        _set_autotune_verbose(verbose)
        reset_sol_attn_stats()
        return io.NodeOutput(m)


class SolAttnVideoBlockProbe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SolAttnVideoBlockProbe",
            display_name="Sol-Attn Video Block Probe",
            is_experimental=True,
            category="sol_attn",
            description="Diagnostic. Place after Patch Sol-Attn: every attention call is "
                        "computed both sparse and dense, and each block's relative error "
                        "is logged worst-first when sampling ends. Paste the top entries "
                        "into the patch node's dense_blocks. The image this produces is "
                        "the dense reference, and the run costs roughly dense + sparse, "
                        "so remove the node once you have the numbers.",
            inputs=[
                io.Model.Input("model"),
                io.String.Input("output_path", default="/tmp/h3_sol_sage_block_probe.json"),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, output_path="/tmp/h3_sol_sage_block_probe.json") -> io.NodeOutput:
        m = model.clone()
        inner = m.model_options["transformer_options"].get("optimized_attention_override")
        if inner is None:
            raise RuntimeError(
                "Sol-Attn Block Probe needs an attention override to measure; "
                "connect it after Patch Sol-Attn.")

        diffusion_model = m.get_model_object("diffusion_model")
        if not _install_block_index(diffusion_model):
            logging.warning(
                f"[sol_attn] probe: {type(diffusion_model).__name__} has no .blocks "
                "list, so every error lands under block 'None'")

        m.model_options["transformer_options"]["optimized_attention_override"] = \
            make_probe_override(inner)
        _probe.clear()
        _probe_timings.clear()
        m.add_callback(
            CallbacksMP.ON_CLEANUP,
            partial(log_probe_summary, output_path=output_path),
        )
        return io.NodeOutput(m)


@wrap_attn
def attention_sol(q, k, v, heads, mask=None, attn_precision=None,
                  skip_reshape=False, skip_output_reshape=False, **kwargs):
    """Registry-visible entry point using defaults (no sigma gating)."""
    if mask is None:
        try:
            out = _run(q, k, v, heads, skip_reshape, skip_output_reshape,
                       kwargs.get("scale", None), 1.0, 4096, False)
            if out is not None:
                return out
        except Exception as exc:
            _stats["errors"] += 1
            _log_kernel_failure(exc)
    return attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
                             skip_output_reshape=skip_output_reshape, **kwargs)


register_attention_function("sol_attn", attention_sol)

if os.environ.get("SOL_ATTN", "0") not in ("0", "", "false"):
    if _sol_attn_kernel is None:
        logging.error(f"[sol_attn] SOL_ATTN set but kernel import failed: {_IMPORT_ERROR}")
    else:
        import comfy.ldm.modules.attention as _attn_mod

        _attn_mod.optimized_attention = attention_sol
        _attn_mod.optimized_attention_masked = attention_sol
        logging.info("[sol_attn] global override active (node patch is preferred)")

class SolAttnVideoExtension(ComfyExtension):
    async def get_node_list(self):
        return [SolAttnVideoPatch, SolAttnVideoBlockProbe]


async def comfy_entrypoint() -> SolAttnVideoExtension:
    return SolAttnVideoExtension()

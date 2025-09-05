# qim/core.py
# One job at a time. No extra process. Keeps the model in memory.
# Target: AMD Strix Halo (ROCm), 128GB RAM → prefer GPU (bf16). No mmap by default.

from __future__ import annotations
import os
import re
import time
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, List, Literal, Tuple, Dict
from contextlib import contextmanager
from threading import Lock

# ---- public types ------------------------------------------------------------

Stage = Literal["model_loading", "pipeline_loading", "lora_loading", "generation"]

ProgressCB = Callable[[Stage, str, Optional[float]], None]
# cb(stage, message, progress 0..1 or None)

# ---- helpers (paths, progress, device) ---------------------------------------

_OUTPUT_DIR = Path.home() / ".qwen-image-studio"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _now_ts() -> str:
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")

def _emit(cb: Optional[ProgressCB], stage: Stage, msg: str, p: Optional[float] = None):
    if not cb:
        return
    try:
        import inspect, asyncio
        if inspect.iscoroutinefunction(cb):
            asyncio.get_running_loop().create_task(cb(stage, msg, p))
        else:
            cb(stage, msg, p)
    except Exception:
        pass
    
def _get_device_and_dtype() -> Tuple[str, "torch.dtype"]:
    import torch
    # Order: MPS → CUDA/ROCm → CPU (bf16 on accel, f32 on CPU)
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32

def _generator_for(device: str, seed: int):
    import torch
    gen_device = "cpu" if device == "mps" else device
    return torch.Generator(device=gen_device).manual_seed(int(seed))

# ---- LoRA utilities (Lightning + custom) -------------------------------------

def _get_lightning_lora_path(ultra_fast: bool) -> Optional[str]:
    # v1.0 (4 steps) or v1.1 (8 steps)
    from huggingface_hub import hf_hub_download
    filename = "Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors" if ultra_fast else \
               "Qwen-Image-Lightning-8steps-V1.1.safetensors"
    try:
        return hf_hub_download(
            repo_id="lightx2v/Qwen-Image-Lightning",
            filename=filename,
            repo_type="model",
        )
    except Exception:
        return None

def _resolve_custom_lora(lora_spec: str) -> Optional[str]:
    """
    Accepts:
      - local '/path/to/file.safetensors' (or '~/Downloads/...'),
      - HF URL 'https://huggingface.co/owner/repo',
      - HF repo id 'owner/repo'.
    Returns local path to a .safetensors file.
    """
    from huggingface_hub import hf_hub_download, list_repo_files
    p = Path(lora_spec).expanduser()
    if p.is_file() and p.suffix == ".safetensors":
        return str(p.resolve())
    repo_id = None
    if lora_spec.startswith("https://huggingface.co/"):
        m = re.match(r"https://huggingface\.co/([^/]+/[^/]+)", lora_spec)
        if m: repo_id = m.group(1)
    else:
        repo_id = lora_spec
    if not repo_id:
        return None
    try:
        files = list_repo_files(repo_id, repo_type="model")
        safes = [f for f in files if f.endswith(".safetensors")]
        if not safes:
            return None
        # prefer names with "lora"
        safes.sort(key=lambda f: ("lora" not in f.lower(), f))
        return hf_hub_download(repo_id=repo_id, filename=safes[0], repo_type="model")
    except Exception:
        return None

def _merge_lora_into_pipe(pipe, lora_path: str, cb: Optional[ProgressCB]) -> None:
    """
    In-place weight merge. Irreversible (reload base to switch).
    Tries several common LoRA formats (diffusers / A-B / .lora.up/.down).
    """
    import safetensors.torch as st
    import torch

    state = st.load_file(lora_path) if hasattr(st, "load_file") else st.load(open(lora_path, "rb").read())
    transformer = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
    if transformer is None:
        raise RuntimeError("Could not find pipe.transformer/unet to merge LoRA into")

    keys = set(state.keys())
    uses_dot = any(".lora.down" in k or ".lora.up" in k for k in keys)
    uses_diff = any(k.startswith("lora_unet_") for k in keys)
    uses_ab  = any(".lora_A" in k or ".lora_B" in k for k in keys)

    def _diff2tx(k: str) -> str:
        k = k.replace("lora_unet_", "")
        k = re.sub(r"transformer_blocks_(\d+)", r"transformer_blocks.\1", k)
        rep = {
            "_attn_add_k_proj": ".attn.add_k_proj",
            "_attn_add_q_proj": ".attn.add_q_proj",
            "_attn_add_v_proj": ".attn.add_v_proj",
            "_attn_to_add_out": ".attn.to_add_out",
            "_ff_context_mlp_fc1": ".ff_context.net.0",
            "_ff_context_mlp_fc2": ".ff_context.net.2",
            "_ff_mlp_fc1": ".ff.net.0",
            "_ff_mlp_fc2": ".ff.net.2",
            "_attn_to_k": ".attn.to_k",
            "_attn_to_q": ".attn.to_q",
            "_attn_to_v": ".attn.to_v",
            "_attn_to_out_0": ".attn.to_out.0",
        }
        for a, b in rep.items(): k = k.replace(a, b)
        return k

    def _device_merge(param, down, up, scaling: float):
        dev = param.device
        up   = up.to(device=dev, dtype=torch.float32)
        down = down.to(device=dev, dtype=torch.float32)
        delta = torch.matmul(up, down) * float(scaling)
        param.data.add_(delta.to(dtype=param.data.dtype))

    # Dry count
    def _count() -> int:
        cnt = 0
        if uses_ab:
            for name, _ in transformer.named_parameters():
                base = name[:-7] if name.endswith(".weight") else name
                a1, b1 = f"diffusion_model.{base}.lora_A.weight", f"diffusion_model.{base}.lora_B.weight"
                a2, b2 = f"{base}.lora_A.weight", f"{base}.lora_B.weight"
                if (a1 in keys and b1 in keys) or (a2 in keys and b2 in keys):
                    cnt += 1
        elif uses_diff:
            bases: Dict[str, set] = {}
            for k in keys:
                if not k.startswith("lora_unet_"): continue
                base = _diff2tx(k.replace(".lora_down.weight","").replace(".lora_up.weight","").replace(".alpha",""))
                bases.setdefault(base, set()).add(k)
            for name, _ in transformer.named_parameters():
                base = name[:-7] if name.endswith(".weight") else name
                ks = bases.get(base)
                if not ks: continue
                has_down = any(k.endswith(".lora_down.weight") for k in ks)
                has_up   = any(k.endswith(".lora_up.weight")   for k in ks)
                if has_down and has_up: cnt += 1
        else:
            for name, _ in transformer.named_parameters():
                base = name[:-7] if name.endswith(".weight") else name
                if uses_dot:
                    kd = f"transformer.{base}.lora.down.weight"
                    ku = f"transformer.{base}.lora.up.weight"
                    if kd not in keys:
                        kd = f"{base}.lora.down.weight"
                        ku = f"{base}.lora.up.weight"
                else:
                    kd = f"{base}.lora_down.weight"
                    ku = f"{base}.lora_up.weight"
                if kd in keys and ku in keys: cnt += 1
        return cnt

    total = max(1, _count())
    merged = 0
    _emit(cb, "lora_loading", "Merging LoRA", 0.0)

    def _bump():
        nonlocal merged
        merged += 1
        _emit(cb, "lora_loading", "Merging LoRA", min(1.0, merged / total))

    # Actual merge
    if uses_ab:
        for name, param in transformer.named_parameters():
            base = name[:-7] if name.endswith(".weight") else name
            for a, b in (
                (f"diffusion_model.{base}.lora_A.weight", f"diffusion_model.{base}.lora_B.weight"),
                (f"{base}.lora_A.weight",                f"{base}.lora_B.weight"),
            ):
                if a in state and b in state:
                    _device_merge(param, state[a], state[b], float(state.get(f"{base}.alpha", 1.0)))
                    _bump()
                    break
    elif uses_diff:
        # collect pairs
        pairs: Dict[str, Tuple[str, str]] = {}
        for k in keys:
            if not k.startswith("lora_unet_"): continue
            base = _diff2tx(k.replace(".lora_down.weight","").replace(".lora_up.weight","").replace(".alpha",""))
            dn = f"lora_unet_{base}.lora_down.weight"
            up = f"lora_unet_{base}.lora_up.weight"
            if dn in keys and up in keys:
                pairs[base] = (dn, up)
        for name, param in transformer.named_parameters():
            base = name[:-7] if name.endswith(".weight") else name
            if base in pairs:
                dn, up = pairs[base]
                _device_merge(param, state[dn], state[up], float(state.get(f"{base}.alpha", 1.0)))
                _bump()
    else:
        for name, param in transformer.named_parameters():
            base = name[:-7] if name.endswith(".weight") else name
            kd, ku = (f"{base}.lora_down.weight", f"{base}.lora_up.weight")
            if uses_dot:
                kd, ku = (f"{base}.lora.down.weight", f"{base}.lora.up.weight")
                if kd not in keys:
                    kd, ku = (f"transformer.{base}.lora.down.weight", f"transformer.{base}.lora.up.weight")
            if kd in keys and ku in keys:
                _device_merge(param, state[kd], state[ku], float(state.get(f"{base}.alpha", 1.0)))
                _bump()

    _emit(cb, "lora_loading", "Merged", 1.0)

# ---- main manager ------------------------------------------------------------

@dataclass
class _LoadedState:
    kind: Literal["generate", "edit"]              # which pipeline type
    lightning: Literal["none", "fast", "ultra"]    # which Lightning LoRA (if any)
    custom_lora_path: Optional[str]                # merged custom LoRA (if any)

class QwenImageManager:
    """
    Singleton-style in-process manager.
    Loads once, serves many jobs. Reloads only when kind/LoRA mode changes.
    """
    def __init__(self, no_mmap: bool = True):
        self._lock = Lock()
        self._pipe = None
        self._state: Optional[_LoadedState] = None
        self._device, self._dtype = _get_device_and_dtype()
        self._no_mmap_enabled = False
        if no_mmap:
            self._enable_no_mmap()

    # -- public ----------------------------------------------------------------

    def warmup(self, cb: Optional[ProgressCB] = None):
        """
        Preload default at startup:
          - Qwen/Qwen-Image (generate)
          - Lightning LoRA v1.0 (4-step) merged
        """
        self._ensure_loaded(
            kind="generate",
            lightning="ultra",
            custom_lora=None,
            cb=cb
        )

    def generate(
        self,
        prompt: str,
        *,
        steps: int = 50,
        seed: Optional[int] = None,
        num_images: int = 1,
        size: str = "16:9",
        fast: bool = False,
        ultra_fast: bool = False,
        lora: Optional[str] = None,
        batman: bool = False,
        cb: Optional[ProgressCB] = None,
    ) -> List[str]:
        """
        Returns list of ABSOLUTE file paths to saved PNGs.
        """
        # Decide mode
        lightning = "ultra" if ultra_fast else ("fast" if fast else "none")
        self._ensure_loaded(kind="generate", lightning=lightning, custom_lora=lora, cb=cb)

        # Resolve steps/cfg based on mode
        if lightning == "ultra":  # 4 steps, cfg 1.0
            num_steps, cfg = 4, 1.0
        elif lightning == "fast": # 8 steps, cfg 1.0
            num_steps, cfg = 8, 1.0
        else:
            num_steps, cfg = int(steps), 4.0

        # Aspect presets (match CLI)
        w, h = {
            "1:1": (1328, 1328), "16:9": (1664, 928), "9:16": (928, 1664),
            "4:3": (1472, 1140), "3:4": (1140, 1472), "3:2": (1584, 1056), "2:3": (1056, 1584)
        }.get(size, (1664, 928))

        # Batman extras (kept for parity)
        additions = [
            " Add a tiny LEGO Batman photobombing from a corner.",
            " Include a LEGO Batman minifigure sneaking into the frame.",
            " A miniature LEGO Batman peeks from the edge.",
            " Tiny LEGO Batman hero pose in the background.",
            " LEGO Batman swings on a small grappling hook.",
        ] if batman else None

        import torch
        pipe = self._pipe
        assert pipe is not None, "Pipeline not loaded"

        _emit(cb, "generation", f"Steps={num_steps}, CFG={cfg}, size={size}, images={num_images}", 0.0)

        timestamp = _now_ts()
        out_paths: List[str] = []
        for i in range(max(1, int(num_images))):
            per_seed = (int(seed) + i) if seed is not None else secrets.randbits(63)
            gen = _generator_for(self._device, per_seed)

            cur_prompt = prompt
            if additions:
                import random as _rnd
                cur_prompt = prompt + _rnd.choice(additions)

            _emit(cb, "generation", f"Invoking pipeline ({i+1}/{num_images})", None)
            # Patch tqdm to surface percent progress through callback
            with _tqdm_patch(lambda pct: _emit(cb, "generation", "denoise", pct / 100.0)):
                image = pipe(
                    prompt=cur_prompt,
                    negative_prompt=" ",
                    width=w, height=h,
                    num_inference_steps=num_steps,
                    true_cfg_scale=cfg,
                    generator=gen,
                ).images[0]

            # Save
            suffix = f"-{i+1}" if num_images > 1 else ""
            out = _OUTPUT_DIR / f"image-{timestamp}{suffix}.png"
            _save_png_with_meta(
                image,
                out,
                {
                    "qim:prompt": cur_prompt,
                    "qim:negative_prompt": " ",
                    "qim:steps": str(num_steps),
                    "qim:cfg_scale": str(cfg),
                    "qim:mode": "ultra-fast" if lightning == "ultra" else ("fast" if lightning == "fast" else "normal"),
                    "qim:seed": str(per_seed),
                    "qim:timestamp": timestamp,
                    "qim:model": "Qwen/Qwen-Image",
                    "qim:size": f"{w}x{h}",
                },
            )
            out_paths.append(str(out.resolve()))

            # progress tick per image
            _emit(cb, "generation", f"Saved {out.name}", (i + 1) / max(1, int(num_images)))

        return out_paths

    def edit(
        self,
        image_path: str,
        prompt: str,
        *,
        steps: int = 50,
        seed: Optional[int] = None,
        fast: bool = False,
        ultra_fast: bool = False,
        lora: Optional[str] = None,
        batman: bool = False,
        output: Optional[str] = None,
        cb: Optional[ProgressCB] = None,
    ) -> List[str]:
        """
        Returns [abs_path_to_png]
        """
        lightning = "ultra" if ultra_fast else ("fast" if fast else "none")
        self._ensure_loaded(kind="edit", lightning=lightning, custom_lora=lora, cb=cb)

        if lightning == "ultra":
            num_steps, cfg = 4, 1.0
        elif lightning == "fast":
            num_steps, cfg = 8, 1.0
        else:
            num_steps, cfg = int(steps), 4.0

        from PIL import Image
        image = Image.open(image_path).convert("RGB")

        seed_val = int(seed) if seed is not None else secrets.randbits(63)
        gen = _generator_for(self._device, seed_val)

        edit_prompt = prompt
        if batman:
            edit_prompt = prompt + " Add a tiny LEGO Batman minifigure photobombing."

        _emit(cb, "generation", f"Editing: steps={num_steps}, CFG={cfg}", 0.0)

        pipe = self._pipe
        assert pipe is not None, "Edit pipeline not loaded"
        with _tqdm_patch(lambda pct: _emit(cb, "generation", "denoise", pct / 100.0)):
            out = pipe(
                image=image,
                prompt=edit_prompt,
                negative_prompt=" ",
                num_inference_steps=num_steps,
                generator=gen,
                guidance_scale=cfg,
            ).images[0]

        # Save
        if output:
            out_path = Path(output)
            if not out_path.is_absolute():
                out_path = _OUTPUT_DIR / output
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = _OUTPUT_DIR / f"edited-{_now_ts()}.png"

        _save_png_with_meta(
            out,
            out_path,
            {
                "qim:prompt": edit_prompt,
                "qim:negative_prompt": " ",
                "qim:steps": str(num_steps),
                "qim:cfg_scale": str(cfg),
                "qim:mode": "ultra-fast" if lightning == "ultra" else ("fast" if lightning == "fast" else "normal"),
                "qim:seed": str(seed_val),
                "qim:model": "Qwen/Qwen-Image-Edit",
            },
        )
        _emit(cb, "generation", f"Saved {out_path.name}", 1.0)
        return [str(out_path.resolve())]

    # -- internal --------------------------------------------------------------

    def _ensure_loaded(
        self,
        *,
        kind: Literal["generate", "edit"],
        lightning: Literal["none", "fast", "ultra"],
        custom_lora: Optional[str],
        cb: Optional[ProgressCB],
    ):
        """
        If current pipeline already matches (kind + lightning + custom_lora), do nothing.
        Otherwise free current, load requested, merge LoRAs on GPU.
        """
        with self._lock:
            # Fast-path: already loaded
            if self._state and \
               self._state.kind == kind and \
               self._state.lightning == lightning and \
               ((self._state.custom_lora_path or None) == (self._resolve_custom(custom_lora) if custom_lora else None)):
                return

            # Free old pipe (keep encoders/vae in RAM is overkill — simple & safe first)
            self._free_current()

            # Load base
            model_name = "Qwen/Qwen-Image-Edit" if kind == "edit" else "Qwen/Qwen-Image"
            _emit(cb, "model_loading", f"Loading base: {model_name}", 0.0)
            if kind == "edit":
                from diffusers import QwenImageEditPipeline as _Pipe
                pipe = _Pipe.from_pretrained(model_name, dtype=self._dtype)
            else:
                from diffusers import DiffusionPipeline as _Pipe
                pipe = _Pipe.from_pretrained(model_name, dtype=self._dtype)

            _emit(cb, "pipeline_loading", "Moving to device", None)
            pipe = pipe.to(self._device)

            # VAE settings (bf16 + tiling if available). Keep on device.
            try:
                if hasattr(pipe, "vae"):
                    pipe.vae.to(device=self._device, dtype=self._dtype)
                    if hasattr(pipe.vae, "enable_tiling"):
                        pipe.vae.enable_tiling()
            except Exception:
                pass

            # Progress bar visible hints
            try:
                pipe.set_progress_bar_config(disable=False, leave=True, miniters=1)
            except Exception:
                pass
            _emit(cb, "pipeline_loading", "Pipeline ready", 1.0)

            # Merge custom LoRA (first)
            custom_path = self._resolve_custom(custom_lora) if custom_lora else None
            if custom_path:
                _merge_lora_into_pipe(pipe, custom_path, cb)

            # Merge Lightning LoRA
            if lightning != "none":
                ultra = (lightning == "ultra")
                lora_path = _get_lightning_lora_path(ultra)
                if lora_path:
                    _merge_lora_into_pipe(pipe, lora_path, cb)

            self._pipe = pipe
            self._state = _LoadedState(kind=kind, lightning=lightning, custom_lora_path=custom_path)

    def _free_current(self):
        if self._pipe is None:
            return
        try:
            # Drop reference and free VRAM
            import torch
            try:
                self._pipe.to("cpu")
            except Exception:
                pass
            self._pipe = None
            self._state = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _resolve_custom(self, lora: Optional[str]) -> Optional[str]:
        if not lora:
            return None
        return _resolve_custom_lora(lora)

    def _enable_no_mmap(self):
        if self._no_mmap_enabled:
            return
        try:
            import safetensors.torch as _st
            from functools import wraps
            old = _st.load_file
            @wraps(old)
            def _no_mmap_load_file(filename, device=None, **kwargs):
                with open(filename, "rb") as f:
                    data = f.read()
                return _st.load(data)  # device kw ignored intentionally
            _st.load_file = _no_mmap_load_file
            self._no_mmap_enabled = True
        except Exception:
            self._no_mmap_enabled = False

# ---- tqdm progress bridge ----------------------------------------------------

@contextmanager
def _tqdm_patch(on_pct: Callable[[int], None]):
    """
    Replaces tqdm so we can get % callbacks from diffusers denoise loop.
    """
    try:
        import tqdm.auto as _tqdm_auto
    except Exception:
        yield
        return

    orig = _tqdm_auto.tqdm

    class _Bridge(orig):  # type: ignore
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._last = -1
        def update(self, n=1):
            r = super().update(n)
            if self.total and self.n is not None:
                pct = int(self.n * 100 // self.total)
                if pct != self._last:
                    self._last = pct
                    try: on_pct(pct)
                    except Exception: pass
            return r

    _tqdm_auto.tqdm = _Bridge
    try:
        yield
    finally:
        _tqdm_auto.tqdm = orig

# ---- image save with PNG metadata -------------------------------------------

def _save_png_with_meta(pil_image, path: Path, meta: Dict[str, str]):
    from PIL.PngImagePlugin import PngInfo
    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngInfo()
    for k, v in (meta or {}).items():
        try: info.add_text(str(k), str(v))
        except Exception: pass
    pil_image.save(str(path), pnginfo=info)

# ---- module-level singleton --------------------------------------------------

_manager: Optional[QwenImageManager] = None

def get_manager() -> QwenImageManager:
    global _manager
    if _manager is None:
        # Default: disable safetensors mmap (ROCm/Strix Halo quirk), preload ultra-fast gen.
        _manager = QwenImageManager(no_mmap=True)
        # Do not auto-warmup here; let caller decide (server can call .warmup()).
    return _manager

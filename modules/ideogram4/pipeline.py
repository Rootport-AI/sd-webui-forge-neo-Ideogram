"""Lazy loader + defensive caller for the official Ideogram4Pipeline (spec §4.3, §4.6).

This module deliberately performs *all* heavy imports (torch, diffusers, the
``ideogram4`` package) lazily inside functions, so that importing this module
(and therefore loading the WebUI UI) never requires the model to be installed.

The exact ``Ideogram4Pipeline.__call__`` signature can differ between the
diffusers integration and the standalone ``ideogram4`` package, and cannot be
verified in this environment.  ``call_pipeline`` therefore introspects the real
signature and only passes keyword arguments the pipeline actually accepts —
unknown args are dropped (with a debug log) rather than crashing generation.
"""

import contextlib
import inspect
import logging
import os

logger = logging.getLogger("ideogram4")

_PIPELINE_CACHE: dict = {}


class Ideogram4Error(RuntimeError):
    """Raised for user-actionable problems (missing package, weights, token, HW)."""


def _import_pipeline_class():
    """Locate the Ideogram4Pipeline class from diffusers or the ideogram4 package."""
    errors = []
    for module, attr in (
        ("ideogram4", "Ideogram4Pipeline"),
        ("ideogram4.pipeline_ideogram4", "Ideogram4Pipeline"),
        ("ideogram4.pipeline", "Ideogram4Pipeline"),
        ("diffusers", "Ideogram4Pipeline"),
    ):
        try:
            mod = __import__(module, fromlist=[attr])
            return getattr(mod, attr)
        except Exception as e:  # ImportError or AttributeError
            errors.append(f"{module}.{attr}: {e}")

    raise Ideogram4Error(
        "Could not import Ideogram4Pipeline. Install the official inference code "
        "(not on PyPI yet):\n"
        "  pip install git+https://github.com/ideogram-oss/ideogram4.git\n"
        "or a diffusers build that ships Ideogram4Pipeline.\nTried:\n  "
        + "\n  ".join(errors)
    )


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _import_config_class():
    """Locate Ideogram4PipelineConfig from the official ideogram4 package (or None)."""
    for module in ("ideogram4", "ideogram4.pipeline_ideogram4", "ideogram4.pipeline"):
        try:
            mod = __import__(module, fromlist=["Ideogram4PipelineConfig"])
        except Exception:
            continue
        cls = getattr(mod, "Ideogram4PipelineConfig", None)
        if cls is not None:
            return cls
    return None


# Used when no explicit model path is given: pick the gated repo per quantization,
# mirroring the official run_inference.py QUANTIZATION_REPOS mapping.
DEFAULT_REPOS = {
    "nf4": "ideogram-ai/ideogram-4-nf4",
    "fp8": "ideogram-ai/ideogram-4-fp8",
}


def _apply_hf_token():
    """Export the configured HF token to the environment so huggingface_hub picks it
    up — the official from_pretrained() has no token argument.

    A token set in Settings is authoritative (so 'Apply settings' is reflected on the
    next generation without a UI reload); when Settings is empty we leave any
    externally-provided HF_TOKEN / HUGGING_FACE_HUB_TOKEN untouched.
    """
    settings_token = ""
    try:
        from modules import shared

        settings_token = (getattr(shared.opts, "ideogram4_hf_token", "") or "").strip()
    except Exception:
        pass
    if settings_token:
        os.environ["HF_TOKEN"] = settings_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = settings_token


def _patch_transformers_extra_special_tokens():
    """Work around an Ideogram-4 tokenizer / Transformers 4.56.x incompatibility.

    `ideogram-ai/ideogram-4-*` ships `tokenizer_config.json` with `extra_special_tokens`
    as a LIST, but Transformers' `PreTrainedTokenizerBase._set_model_specific_special_tokens`
    treats it as a mapping (calls `.keys()`/`.items()`) and crashes with
    `AttributeError: 'list' object has no attribute 'keys'` before the pipeline finishes
    loading. Those tokens already exist in the Qwen vocab, so when a list is received we
    simply skip the model-specific special-token map step.

    Idempotent, applied in-process only (no cache/site-packages files are modified).
    """
    try:
        from transformers import PreTrainedTokenizerBase
    except Exception:
        return

    if getattr(PreTrainedTokenizerBase, "_ideogram4_extra_special_tokens_patch", False):
        return

    original = getattr(PreTrainedTokenizerBase, "_set_model_specific_special_tokens", None)
    if original is None:
        return

    def patched(self, special_tokens, *args, **kwargs):
        if isinstance(special_tokens, list):
            return
        return original(self, special_tokens, *args, **kwargs)

    PreTrainedTokenizerBase._set_model_specific_special_tokens = patched
    PreTrainedTokenizerBase._ideogram4_extra_special_tokens_patch = True
    logger.debug("Applied Ideogram 4.0 tokenizer extra_special_tokens compatibility patch")


@contextlib.contextmanager
def _offline_env(enabled: bool):
    """Temporarily force HF / transformers offline for the duration of the load, then
    restore the previous environment.

    Only touches the environment when ``enabled`` (i.e. the UI checkbox). Any value
    that was already present (e.g. set externally by the user) is saved and restored,
    so unchecking the box never leaves a stale offline state behind — fixing the bug
    where a failed offline run kept failing after the box was turned off.
    """
    if not enabled:
        yield
        return

    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ[k] = "1"
        yield
    finally:
        for k in keys:
            old = previous[k]
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _check_transformers_for_ideogram4():
    """Gate generation when Transformers is too old for Qwen3-VL (spec: Ideogram4 mode).

    Does NOT install anything — the Transformers swap happens at startup
    (modules_forge.ideogram4_transformers_mode) based on the saved UI preset.
    """
    try:
        import importlib.metadata

        ver = importlib.metadata.version("transformers")
    except Exception:
        ver = None

    too_old = False
    if ver is not None:
        try:
            from packaging.version import Version

            too_old = Version(ver) < Version("4.57.1")
        except Exception:
            too_old = False

    qwen_ok = True
    try:
        import transformers.models.qwen3_vl  # noqa: F401
    except Exception:
        qwen_ok = False

    if too_old or not qwen_ok:
        raise Ideogram4Error(
            "Ideogram 4.0 requires Transformers 4.57.1 or newer because its text encoder "
            "uses Qwen3-VL.\n"
            f"This Forge Neo process is currently running Transformers {ver}.\n\n"
            "Please select the Ideogram4 UI preset, then fully restart Forge Neo from "
            "webui-user.bat/webui.bat.\n"
            "Settings -> Reload UI is not enough because Python packages are already "
            "imported in the current process."
        )


def get_pipeline(model_path: str, quantization: str = "nf4", offline_mode: bool = False):
    """Load (and cache) the Ideogram4Pipeline.

    ``model_path`` is a Hugging Face repo id (e.g. ``ideogram-ai/ideogram-4-nf4``)
    or anything the official loader accepts as ``weights_repo``; if empty it
    defaults to the gated repo for the chosen ``quantization`` (nf4 / fp8).

    ``offline_mode`` forces huggingface_hub / transformers to use only the local
    cache (no network); enable it once all required files are cached.

    The official ``Ideogram4Pipeline.from_pretrained`` is keyword-only and takes a
    ``config=Ideogram4PipelineConfig(weights_repo=...)`` plus ``device`` / ``dtype``
    (NOT a diffusers-style positional path). Raises ``Ideogram4Error`` with an
    actionable message on misconfiguration.
    """
    quantization = (quantization or "nf4").lower()
    weights_repo = model_path or DEFAULT_REPOS.get(quantization, DEFAULT_REPOS["nf4"])

    # offline_mode is intentionally NOT part of the cache key: once loaded the
    # pipeline runs locally regardless, and a cached pipe is reused either way.
    cache_key = (weights_repo, quantization)
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    _check_transformers_for_ideogram4()

    if not _cuda_available():
        raise Ideogram4Error(
            "Ideogram 4.0 requires a CUDA GPU, which is not available here. "
            "Run on a machine with an NVIDIA GPU (both nf4 and fp8 weights need CUDA)."
        )

    PipelineClass = _import_pipeline_class()
    ConfigClass = _import_config_class()
    _apply_hf_token()
    _patch_transformers_extra_special_tokens()

    import torch

    logger.info("Loading Ideogram 4.0 pipeline from %s (%s, offline=%s)", weights_repo, quantization, offline_mode)
    try:
        with _offline_env(offline_mode):
            if ConfigClass is not None:
                config = ConfigClass(weights_repo=weights_repo)
                pipe = PipelineClass.from_pretrained(config=config, device="cuda", dtype=torch.bfloat16)
            else:
                # Fallback for a diffusers-style Ideogram4Pipeline (positional repo/path).
                pipe = PipelineClass.from_pretrained(weights_repo)
                if hasattr(pipe, "to"):
                    pipe = pipe.to("cuda")
    except Ideogram4Error:
        raise
    except Exception as e:
        msg = str(e)
        name = type(e).__name__
        lower = msg.lower()
        if offline_mode and ("offline" in name.lower() or "localentrynotfound" in name.lower() or "offline" in lower or ("cache" in lower and ("cannot" in lower or "not found" in lower or "no such" in lower))):
            raise Ideogram4Error(
                "Ideogram 4.0 offline mode is enabled, but required weights are not "
                "available in the local Hugging Face cache. Disable offline mode and run "
                "once online after accepting the Hugging Face license gate."
            ) from e
        if "Gated" in name or "gated" in msg.lower() or "401" in msg or "403" in msg:
            raise Ideogram4Error(
                "Access to the Ideogram 4.0 weights was denied. Accept the license at "
                f"https://huggingface.co/{weights_repo} and set an HF token "
                "(Settings → 'Ideogram 4.0' → 'HF token', or the HF_TOKEN env var)."
            ) from e
        raise

    _PIPELINE_CACHE[cache_key] = pipe
    return pipe


def clear_cache():
    _PIPELINE_CACHE.clear()


def _build_generator(seed):
    if seed is None or seed < 0:
        return None
    try:
        import torch

        device = "cuda" if _cuda_available() else "cpu"
        return torch.Generator(device=device).manual_seed(int(seed))
    except Exception:
        logger.warning("Could not build a torch.Generator for seed %s", seed, exc_info=True)
        return None


def call_pipeline(
    pipe,
    prompt: str,
    *,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    guidance_schedule=None,
    mu=None,
    std=None,
    negative_prompt=None,
    transparent: bool = False,
    seed=None,
    num_images: int = 1,
):
    """Call the pipeline, passing only kwargs its ``__call__`` actually accepts.

    Returns a list of PIL images.
    """
    try:
        sig = inspect.signature(pipe.__call__)
        accepted = set(sig.parameters)
        has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except (TypeError, ValueError):
        accepted = set()
        has_var_kw = True

    def pick(*names):
        """Return the first accepted parameter name, or the canonical one under **kwargs."""
        for n in names:
            if n in accepted:
                return n
        return names[0] if has_var_kw else None

    generator = _build_generator(seed)

    # logical arg -> (candidate parameter names in priority order, value)
    logical = [
        (("height",), height),
        (("width",), width),
        (("num_steps", "num_inference_steps"), steps),
        (("guidance_scale",), guidance_scale),
        (("guidance_schedule",), guidance_schedule),
        (("mu",), mu),
        (("std",), std),
        (("negative_prompt",), negative_prompt),
        (("transparent", "transparent_background"), True if transparent else None),
        (("num_images_per_prompt", "num_images"), num_images),
        # official pipeline runs its own caption verifier and raises by default;
        # we surface warnings ourselves (spec §4.5), so never let it block generation
        (("raise_on_caption_issues",), False),
    ]

    kwargs = {}
    for names, value in logical:
        if value is None:
            continue
        key = pick(*names)
        if key is not None:
            kwargs[key] = value

    # seed: prefer a real generator, else fall back to a `seed`/`generator` kwarg
    if generator is not None and (("generator" in accepted) or has_var_kw):
        kwargs["generator"] = generator
    elif seed is not None and seed >= 0:
        key = pick("seed", "generator")
        if key == "seed":
            kwargs["seed"] = int(seed)

    logger.debug("Ideogram4 __call__ kwargs: %s", sorted(kwargs))

    result = pipe(prompt, **kwargs)

    images = getattr(result, "images", None)
    if images is None:
        images = result if isinstance(result, (list, tuple)) else [result]
    return list(images)

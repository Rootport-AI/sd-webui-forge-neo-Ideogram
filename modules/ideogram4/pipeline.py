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
    up — the official from_pretrained() has no token argument."""
    token = None
    try:
        from modules import shared

        token = getattr(shared.opts, "ideogram4_hf_token", None)
    except Exception:
        pass
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


def get_pipeline(model_path: str, quantization: str = "nf4"):
    """Load (and cache) the Ideogram4Pipeline.

    ``model_path`` is a Hugging Face repo id (e.g. ``ideogram-ai/ideogram-4-nf4``)
    or anything the official loader accepts as ``weights_repo``; if empty it
    defaults to the gated repo for the chosen ``quantization`` (nf4 / fp8).

    The official ``Ideogram4Pipeline.from_pretrained`` is keyword-only and takes a
    ``config=Ideogram4PipelineConfig(weights_repo=...)`` plus ``device`` / ``dtype``
    (NOT a diffusers-style positional path). Raises ``Ideogram4Error`` with an
    actionable message on misconfiguration.
    """
    quantization = (quantization or "nf4").lower()
    weights_repo = model_path or DEFAULT_REPOS.get(quantization, DEFAULT_REPOS["nf4"])

    cache_key = (weights_repo, quantization)
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    if not _cuda_available():
        raise Ideogram4Error(
            "Ideogram 4.0 requires a CUDA GPU, which is not available here. "
            "Run on a machine with an NVIDIA GPU (both nf4 and fp8 weights need CUDA)."
        )

    PipelineClass = _import_pipeline_class()
    ConfigClass = _import_config_class()
    _apply_hf_token()

    import torch

    logger.info("Loading Ideogram 4.0 pipeline from %s (%s)", weights_repo, quantization)
    try:
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
        if "Gated" in type(e).__name__ or "gated" in msg.lower() or "401" in msg or "403" in msg:
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

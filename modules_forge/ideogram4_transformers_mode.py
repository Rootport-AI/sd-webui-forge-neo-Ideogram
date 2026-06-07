"""Startup runtime-dependency preflight for the Ideogram 4.0 UI preset.

Ideogram 4.0 needs a specific set of Python packages that the rest of Forge Neo does
not (Qwen3-VL support from a newer Transformers, bitsandbytes for the nf4 weights, and
the official ``ideogram4`` inference code which is not on PyPI). Rather than upgrading
the whole stack — which could affect SDXL / Anima / Flux / Qwen-Image — these are
checked (and, if missing, installed into the current venv) ONLY at process start and
ONLY when the saved ``forge_preset`` is ``ideogram4``.

Scope: Python packages only. Model weights / text encoder / VAE / tokenizer / HF
license-gate / HF token remain the loader's responsibility at generation time
(``modules/ideogram4/pipeline.py``). This split keeps "dependency missing" and "model
download failed" as clearly separate, separately-logged problems.

Must run before transformers / torch are imported into the main process (called from
``launch.py`` after ``prepare_environment()`` and before ``start()``); switching the
preset therefore needs a FULL restart, not Settings -> Reload UI. All verification is
done in subprocesses so this module never imports torch / transformers / bitsandbytes /
ideogram4 into the main process.
"""

import json
import os
import subprocess
import sys
import time

STANDARD_TRANSFORMERS = "4.56.2"
IDEOGRAM4_TRANSFORMERS = "4.57.6"
IDEOGRAM4_PACKAGE_SPEC = "git+https://github.com/ideogram-oss/ideogram4.git"

LOG_TRANSFORMERS = "[Ideogram4/transformers]"
LOG_RUNTIME = "[Ideogram4/runtime]"

PREFLIGHT_LOCK = "ideogram4_runtime_preflight.lock"
_LOCK_STALE_SECONDS = 1800

# subprocess verification snippets (run with `python -c`)
_VERIFY_IDEOGRAM4_TRANSFORMERS = (
    "import transformers, transformers.models.qwen3_vl\n"
    "from transformers.masking_utils import create_causal_mask\n"
)
_VERIFY_STANDARD_TRANSFORMERS = (
    "import transformers\n"
    "from transformers.modeling_utils import no_init_weights\n"
)
_VERIFY_BITSANDBYTES = "import bitsandbytes, bitsandbytes.nn\n"
_VERIFY_IDEOGRAM4 = "from ideogram4 import Ideogram4Pipeline\n"
_VERIFY_BASE = "import accelerate, diffusers, huggingface_hub, safetensors\n"

# Ordered: transformers must be in place before bitsandbytes, and both before the
# ideogram4 import-check (importing Ideogram4Pipeline pulls transformers + bitsandbytes).
IDEOGRAM4_RUNTIME_REQUIREMENTS = [
    {
        "name": "transformers",
        "pip_spec": f"transformers=={IDEOGRAM4_TRANSFORMERS}",
        "install_args": "--no-deps",
        "verify": _VERIFY_IDEOGRAM4_TRANSFORMERS,
        "log_prefix": LOG_TRANSFORMERS,
    },
    {
        "name": "bitsandbytes",
        "pip_spec": "bitsandbytes==0.49.2",
        "install_args": "--no-deps",
        "verify": _VERIFY_BITSANDBYTES,
        "log_prefix": LOG_RUNTIME,
    },
    {
        "name": "ideogram4",
        "pip_spec": IDEOGRAM4_PACKAGE_SPEC,
        "install_args": "--no-deps",
        "verify": _VERIFY_IDEOGRAM4,
        "log_prefix": LOG_RUNTIME,
    },
]


def _log(prefix: str, msg: str):
    print(f"{prefix} {msg}")


def _read_preset(settings_file: str):
    try:
        with open(settings_file, "r", encoding="utf-8") as f:
            return json.load(f).get("forge_preset")
    except FileNotFoundError:
        return None
    except Exception as e:
        _log(LOG_RUNTIME, f"could not read settings file {settings_file!r}: {e}")
        return None


def _installed_version(package: str):
    import importlib.metadata

    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _verify(code: str):
    """Run a verification snippet in a fresh subprocess. Returns (ok, stderr)."""
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    except Exception as e:
        return False, str(e)
    return result.returncode == 0, (result.stderr or "").strip()


def _pip_install(req) -> bool:
    from modules import launch_utils

    try:
        launch_utils.run_pip(
            f"install {req['install_args']} {req['pip_spec']}",
            f"{req['name']} (Ideogram 4.0 runtime)",
        )
        return True
    except Exception as e:
        _log(req.get("log_prefix", LOG_RUNTIME), f"pip install failed for {req['name']}: {e}")
        return False


# ---- lock -----------------------------------------------------------------
def _lock_path(name: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp = os.path.join(root, "tmp")
    try:
        os.makedirs(tmp, exist_ok=True)
    except OSError:
        pass
    return os.path.join(tmp, name)


def _acquire_lock(name: str):
    """Return ``(proceed, path)``; ``path`` is set only when we own the lock file."""
    path = _lock_path(name)
    try:
        if os.path.exists(path) and (time.time() - os.path.getmtime(path)) > _LOCK_STALE_SECONDS:
            _log(LOG_RUNTIME, "removing stale preflight lock")
            os.remove(path)
    except OSError:
        pass
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True, path
    except FileExistsError:
        return False, None
    except OSError as e:
        _log(LOG_RUNTIME, f"could not create lock ({e}); proceeding without it")
        return True, None


def _release_lock(path):
    if path:
        try:
            os.remove(path)
        except OSError:
            pass


# ---- per-requirement handling --------------------------------------------
def _ensure_requirement(req, skip_install: bool) -> bool:
    prefix = req.get("log_prefix", LOG_RUNTIME)
    manual = f"python -m pip install {req['install_args']} {req['pip_spec']}"

    ok, _ = _verify(req["verify"])
    if ok:
        _log(prefix, f"{req['name']} already available")
        return True

    if skip_install:
        _log(prefix, f"--skip-install is set: missing {req['name']}. Ideogram 4.0 generation will fail until you install it manually:\n    {manual}")
        return False

    _log(prefix, f"installing {req['name']} ({req['pip_spec']}) ...")
    if not _pip_install(req):
        _log(prefix, f"could not install {req['name']} — check network / proxy / git / --skip-install, or install manually:\n    {manual}")
        return False

    import importlib

    importlib.invalidate_caches()
    ok2, err = _verify(req["verify"])
    if ok2:
        _log(prefix, f"{req['name']} installed and verified")
    else:
        _log(prefix, f"{req['name']} installed but verification failed (generation may error):\n{err}")
    return ok2


def _final_verify():
    ok, err = _verify(_VERIFY_IDEOGRAM4_TRANSFORMERS + _VERIFY_BITSANDBYTES + _VERIFY_IDEOGRAM4)
    if ok:
        _log(LOG_RUNTIME, "all Ideogram 4.0 runtime dependencies verified")
    else:
        _log(LOG_RUNTIME, f"runtime dependency verification still failing (generation may error):\n{err}")

    base_ok, base_err = _verify(_VERIFY_BASE)
    if not base_ok:
        _log(LOG_RUNTIME, f"Forge Neo base requirements look missing/broken (accelerate / diffusers / huggingface_hub / safetensors); the base environment install may have failed:\n{base_err}")


def _revert_transformers_to_standard(args):
    """Non-ideogram4 presets run on the standard transformers; revert if needed."""
    current = _installed_version("transformers")
    if current == STANDARD_TRANSFORMERS:
        return

    _log(LOG_TRANSFORMERS, f"preset is not ideogram4; transformers is {current}, restoring {STANDARD_TRANSFORMERS}")
    if args.skip_install:
        _log(LOG_TRANSFORMERS, f"--skip-install is set: not reverting (current {current})")
        return

    proceed, lock_path = _acquire_lock(PREFLIGHT_LOCK)
    if not proceed:
        _log(LOG_TRANSFORMERS, "another process is updating dependencies; skipping revert")
        return
    try:
        if _installed_version("transformers") == STANDARD_TRANSFORMERS:
            return
        if not _pip_install({"name": "transformers", "pip_spec": f"transformers=={STANDARD_TRANSFORMERS}", "install_args": "--no-deps", "log_prefix": LOG_TRANSFORMERS}):
            _log(LOG_TRANSFORMERS, f"failed to restore transformers=={STANDARD_TRANSFORMERS}; continuing with {current}")
            return
        ok, err = _verify(_VERIFY_STANDARD_TRANSFORMERS)
        if ok:
            _log(LOG_TRANSFORMERS, f"restored transformers {STANDARD_TRANSFORMERS}")
        else:
            _log(LOG_TRANSFORMERS, f"revert verification failed; continuing:\n{err}")
    finally:
        _release_lock(lock_path)


# ---- entry point ----------------------------------------------------------
def ensure_ideogram4_runtime():
    """Preflight Ideogram 4.0 Python dependencies (or revert transformers for others)."""
    try:
        _run_preflight()
    except Exception as e:
        _log(LOG_RUNTIME, f"unexpected error during preflight: {e!r}; continuing startup")


def _run_preflight():
    from modules import launch_utils

    args = launch_utils.args
    preset = _read_preset(args.ui_settings_file)

    if preset != "ideogram4":
        _revert_transformers_to_standard(args)
        return

    proceed, lock_path = _acquire_lock(PREFLIGHT_LOCK)
    if not proceed:
        _log(LOG_RUNTIME, "another process is running the Ideogram 4.0 preflight; skipping")
        return
    try:
        for req in IDEOGRAM4_RUNTIME_REQUIREMENTS:
            _ensure_requirement(req, args.skip_install)
        _final_verify()
    finally:
        _release_lock(lock_path)


# Backwards-compatible aliases (older launch.py / external callers).
ensure_ideogram4_transformers_mode = ensure_ideogram4_runtime


def ensure_ideogram4_package():
    """Deprecated: folded into ensure_ideogram4_runtime(); kept as a no-op shim."""

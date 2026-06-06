"""Startup-time Transformers version switch for the Ideogram 4.0 UI preset.

Ideogram 4.0's text encoder is Qwen3-VL, which is only recognized by Transformers
>= 4.57.x, whereas the rest of Forge Neo targets ``transformers==4.56.2``. Instead
of upgrading the whole stack (which could affect SDXL / Anima / Flux / Qwen-Image),
we switch Transformers ONLY at process start, based on the saved ``forge_preset``
in ``config.json``:

    forge_preset == "ideogram4"  ->  transformers==4.57.6
    otherwise                    ->  transformers==4.56.2

Python cannot safely swap an already-imported package in-process, and Settings ->
Reload UI reuses the same process, so this MUST run before ``transformers`` is first
imported — it is invoked from ``launch.py`` after ``prepare_environment()`` and just
before ``start()``. Changing the preset therefore needs a FULL restart of Forge Neo
(from webui-user.bat / webui.bat), not Reload UI.

Deliberately lightweight: it imports only json / os / subprocess / sys / time /
importlib.metadata (+ launch_utils for run_pip). It never imports transformers,
torch, gradio, modules.shared, or modules.ideogram4.
"""

import json
import os
import subprocess
import sys
import time

STANDARD_VERSION = "4.56.2"
IDEOGRAM4_VERSION = "4.57.6"
MIN_IDEOGRAM4_VERSION = "4.57.1"

_PREFIX = "[Ideogram4/transformers]"
_LOCK_STALE_SECONDS = 600


def _log(msg: str):
    print(f"{_PREFIX} {msg}")


def _installed_version():
    import importlib.metadata

    try:
        return importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return None


def _read_preset(settings_file: str):
    try:
        with open(settings_file, "r", encoding="utf-8") as f:
            return json.load(f).get("forge_preset")
    except FileNotFoundError:
        return None
    except Exception as e:
        _log(f"could not read settings file {settings_file!r}: {e}")
        return None


def _lock_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp = os.path.join(root, "tmp")
    try:
        os.makedirs(tmp, exist_ok=True)
    except OSError:
        pass
    return os.path.join(tmp, "ideogram4_transformers_switch.lock")


def _acquire_lock():
    """Return ``(proceed, path)``. ``path`` is set only when we own the lock file."""
    path = _lock_path()
    try:
        if os.path.exists(path) and (time.time() - os.path.getmtime(path)) > _LOCK_STALE_SECONDS:
            _log("removing stale switch lock")
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
        _log(f"could not create lock ({e}); proceeding without it")
        return True, None


def _release_lock(path):
    if path:
        try:
            os.remove(path)
        except OSError:
            pass


def _pip_switch(version: str) -> bool:
    from modules import launch_utils

    try:
        # --no-deps: only move Transformers itself (tokenizers / huggingface-hub already
        # satisfy 4.57.6); run_pip adds --prefer-binary, honours INDEX_URL and --uv.
        launch_utils.run_pip(
            f"install --no-deps transformers=={version}",
            f"transformers=={version} (Ideogram 4.0 mode switch)",
        )
        return True
    except Exception as e:
        _log(f"pip switch to transformers=={version} failed: {e}")
        return False


def _verify(ideogram: bool) -> bool:
    """Verify the switched Transformers in a SEPARATE process (never import it here)."""
    if ideogram:
        code = (
            "import transformers, transformers.models.qwen3_vl\n"
            "from transformers.masking_utils import create_causal_mask\n"
            "print(transformers.__version__)\n"
        )
    else:
        code = (
            "import transformers\n"
            "from transformers.modeling_utils import no_init_weights\n"
            "print(transformers.__version__)\n"
        )
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    except Exception as e:
        _log(f"verification subprocess error: {e}")
        return False
    if result.returncode != 0:
        _log("verification failed:\n" + (result.stderr or "").strip())
        return False
    _log(f"verified transformers {(result.stdout or '').strip()}")
    return True


def _on_failure(ideogram: bool, target: str, current):
    if ideogram:
        _log(
            f"FAILED to switch to transformers=={target} (current {current}). Ideogram 4.0 "
            "generation will be blocked by the runtime check. Verify: internet connection, pip "
            "cache, that --skip-install is not set, and antivirus file locks; then fully restart."
        )
    else:
        _log(
            f"FAILED to revert to transformers=={target} (current {current}). Continuing startup "
            "with the current version; if existing models misbehave, fully restart or repair the "
            "environment manually."
        )


def ensure_ideogram4_transformers_mode():
    """Switch Transformers to match the saved UI preset, before transformers is imported."""
    try:
        _ensure()
    except Exception as e:
        _log(f"unexpected error during transformers mode check: {e!r}; continuing startup")


def _ensure():
    from modules import launch_utils

    args = launch_utils.args
    preset = _read_preset(args.ui_settings_file)
    want_ideogram = preset == "ideogram4"
    target = IDEOGRAM4_VERSION if want_ideogram else STANDARD_VERSION
    current = _installed_version()

    if current == target:
        _log(f"transformers=={current} already matches preset {preset!r}")
        return

    _log(f"preset {preset!r} wants transformers=={target}, found {current}")

    if args.skip_install:
        _log(
            f"--skip-install is set: NOT switching transformers (Ideogram 4.0 needs "
            f">= {MIN_IDEOGRAM4_VERSION}). The runtime check will stop generation if incompatible."
        )
        return

    proceed, lock_path = _acquire_lock()
    if not proceed:
        _log("another process is switching transformers; skipping")
        return

    try:
        current = _installed_version()  # re-check under the lock
        if current == target:
            _log(f"transformers=={current} already matches after lock; nothing to do")
            return

        if not _pip_switch(target):
            _on_failure(want_ideogram, target, current)
            return

        if not _verify(want_ideogram):
            _on_failure(want_ideogram, target, _installed_version())
            return

        _log(f"switched transformers to {target} for preset {preset!r}")
    finally:
        _release_lock(lock_path)

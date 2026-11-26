"""Run extension code without write access to host repositories or network."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from .errors import LiftError


ROOT = Path(__file__).resolve().parents[3]
EXTENSIONS = ROOT / "src/intentlift/extensions"


def compiler_worker(request):
    if not shutil.which("bwrap"):
        raise LiftError("infrastructure_failure", "bubblewrap required; no unsafe fallback")
    command = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev",
               "--proc", "/proc", "--tmpfs", str(Path('/tmp').resolve()),
               "--unshare-net", "--unshare-pid", "--cap-drop", "ALL",
               "--die-with-parent", "--new-session", "--clearenv"]
    # Host temporary inputs would otherwise be hidden by the private /tmp.
    temporary_root = Path(tempfile.gettempdir()).resolve()
    inputs = [request.get('extension_root'), request.get('source'), *request.get('include_dirs', [])]
    for name in dict.fromkeys(value for value in inputs if value):
        path = Path(name).resolve(strict=True)
        if path.is_relative_to(temporary_root):
            command += ['--ro-bind', str(path), str(path)]
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp",
           "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1",
           "TMPDIR": "/tmp", "LANG": "C.UTF-8"}
    for name, value in env.items():
        command += ["--setenv", name, value]
    command += [sys.executable, "-m", "intentlift.core.worker"]
    completed = subprocess.run(command, input=json.dumps(request), text=True, capture_output=True)
    if completed.returncode:
        raise LiftError("infrastructure_failure", "compiler sandbox failed", diagnostic=completed.stderr[-2000:])
    try:
        return json.loads(completed.stdout)
    except ValueError as exc:
        raise LiftError("extension_invalid", "compiler worker did not return valid JSON") from exc

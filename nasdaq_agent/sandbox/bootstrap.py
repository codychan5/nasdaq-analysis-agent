"""The bootstrap that runs INSIDE the sandbox: it preloads the data, hardens the interpreter, then
executes the model's analysis code and prints the result on a sentinel line.

Defence in depth only. The Docker backend (a network-less, read-only container) is the real isolation
boundary; this bootstrap is the backstop for the subprocess backend, which runs in the host interpreter.
It disables the network before loading the data. After the data is loaded it wraps builtins.open/io.open,
then rebinds numpy's DataSource opener to the same guard (since numpy captured the built-in open at import),
so file access through those openers -- including numpy's own file readers -- is refused outside the sandbox
and the interpreter's own installation, and refused for every write, even if the static gate (sandbox/gate.py)
missed a name. It is not a substitute for the container.
"""
from pathlib import Path

from .contract import SENTINEL, STDOUT_CAP_CHARS

BOOTSTRAP_FILENAME = "bootstrap.py"
CODE_FILENAME = "analysis_code.py"

# Runs INSIDE the sandbox. Keep it dependency-light; it only needs pandas and the standard library.
BOOTSTRAP_SOURCE = f'''
import builtins, io, json, os, random, site, socket, sys
from pathlib import Path
import numpy as np
import pandas as pd

SENTINEL = {SENTINEL!r}
random.seed(0); np.random.seed(0)

def _no_network(*a, **k):
    raise RuntimeError("network access is disabled in the sandbox")
socket.socket = _no_network
socket.create_connection = _no_network

meta = json.load(open("meta.json"))
df = pd.read_csv("ticker.csv", parse_dates=["date"])
bench = pd.read_csv("benchmark.csv", parse_dates=["date"])
code = open(sys.argv[1]).read()

# With the data already loaded, restrict open() before running the model's code. Writes are
# refused outright; reads are allowed only under the sandbox directory or the interpreter's own
# installation (so lazy library imports and package data keep working). Docker is the real boundary.
_sandbox_root = Path(sys.argv[1]).resolve().parent
_read_roots = [_sandbox_root]
for _p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
    if _p:
        _read_roots.append(Path(_p).resolve())
try:
    for _sp in site.getsitepackages():
        _read_roots.append(Path(_sp).resolve())
except Exception:
    pass
_real_open = builtins.open

def _guarded_open(file, mode="r", *args, **kwargs):
    if not isinstance(mode, str) or any(_c in mode for _c in ("w", "a", "x", "+")):
        raise PermissionError("sandbox: writing files is disabled")
    try:
        _resolved = Path(os.fspath(file)).resolve()
    except TypeError:
        raise PermissionError("sandbox: only path-like reads are allowed")
    if not any(_resolved == _r or _resolved.is_relative_to(_r) for _r in _read_roots):
        raise PermissionError("sandbox: reading outside the sandbox is disabled: " + str(_resolved))
    return _real_open(file, mode, *args, **kwargs)

builtins.open = _guarded_open
io.open = _guarded_open

# numpy's DataSource captured the built-in open at import time
# (numpy/lib/_datasource.py: self._file_openers = {{None: open}}), before this wrapping, so
# fromregex/loadtxt/genfromtxt and numpy.lib._datasource.open would keep using the unguarded open
# even after the lines above. Rebind its default opener to the guarded one and clear its cached
# (compression) openers so they are re-derived through the now-wrapped modules on next use.
try:
    import numpy.lib._datasource as _ds
    _ds._file_openers._file_openers = {{None: _guarded_open}}
    _ds._file_openers._loaded = False
except Exception:
    pass

namespace = {{"df": df, "bench": bench, "meta": meta, "pd": pd, "np": np}}
exec(compile(code, "<analysis>", "exec"), namespace)
result = namespace.get("result")
if not isinstance(result, dict):
    raise SystemExit("contract violation: assign a dict named result")
missing = [k for k in meta["required_keys"] if k not in result]
if missing:
    raise SystemExit("contract violation: result is missing keys " + ", ".join(missing))
line = SENTINEL + json.dumps(result, default=str)
sys.stdout.write(line[:{STDOUT_CAP_CHARS}] + "\\n")
'''


def write_sandbox_files(sandbox_dir: Path, code: str) -> tuple[Path, Path]:
    """Write the bootstrap and the model's code next to the input files."""
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_path = sandbox_dir / BOOTSTRAP_FILENAME
    code_path = sandbox_dir / CODE_FILENAME
    bootstrap_path.write_text(BOOTSTRAP_SOURCE)
    code_path.write_text(code)
    return bootstrap_path, code_path

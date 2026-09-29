"""Static, best-effort screen for careless or hostile analysis code before it runs.

This module is NOT the security boundary: the Docker backend (sandboxed container execution) is
what actually isolates untrusted code. A gap in this gate is a missed fast-fail, not a sandbox
escape -- but every gap closed here is one less case that reaches the runtime defenses at all.
"""
import ast
from dataclasses import dataclass, field

ALLOWED_IMPORTS = {"pandas", "numpy", "math", "statistics", "json", "datetime"}
BANNED_CALLS = {"eval", "exec", "compile", "open", "getattr", "setattr", "delattr", "globals", "locals",
                "vars", "input", "breakpoint", "__import__", "exit", "quit", "help"}
# Attribute names banned outright, regardless of prefix rules below: assorted dangerous methods,
# plus extra readers that don't start with "read_" -- numpy's array/binary loaders and pandas'/
# numpy's IO-source classes -- and module-path components that reach into pandas'/numpy's IO
# subsystems by attribute chain (e.g. pd.io.parsers, np.lib.npyio) rather than a single named call.
# fromregex reads an arbitrary file. (dump/dumps are handled separately -- see JSON_SERIALISER_ATTRIBUTES
# below -- because json.dump/json.dumps are legitimate on the allowed json module.)
BANNED_ATTRIBUTES = {"query", "eval", "system", "popen",
                     "load", "loadtxt", "genfromtxt", "fromfile", "fromregex", "memmap",
                     "DataSource", "ExcelFile", "ExcelWriter", "HDFStore",
                     "io", "lib"}
# dump/dumps are banned on any receiver EXCEPT the json module. ndarray.dump(path)
# writes a file and ndarray.dumps() pickles, but json.dumps returns a string and json.dump needs an
# already-open writable file (which the bootstrap open() guard refuses), so both json forms are safe.
JSON_SERIALISER_ATTRIBUTES = {"dump", "dumps"}
# pandas/numpy file writers, named explicitly rather than banned by a "to_" prefix so that
# converters such as to_numpy, to_list, to_dict and ndarray.tolist keep working -- only methods
# that write to a file or an external IO sink are banned here. to_string/to_latex/to_markdown/to_html
# are NOT here: their no-argument forms return strings and stay allowed; the call check below rejects
# only the forms that pass a file/buffer (a positional argument or a buf= keyword).
BANNED_WRITER_ATTRIBUTES = {"to_csv", "to_json", "to_parquet", "to_excel", "to_feather",
                           "to_hdf", "to_sql", "to_stata", "to_pickle",
                           "to_xml", "to_orc", "to_clipboard", "to_gbq", "tofile",
                           "save", "savetxt", "savez", "savez_compressed"}
# These render to a string when called with no argument (allowed), but write to a file or buffer
# when given a positional argument or a buf= keyword (rejected). DataFrame.to_string("../x") writes a file.
BUFFERED_RENDER_ATTRIBUTES = {"to_string", "to_latex", "to_markdown", "to_html"}
# Any attribute access starting with one of these prefixes is banned. pandas has many read_*
# loaders (read_csv, read_pickle, read_json, read_excel, ...); the sandbox preloads all the data
# analysis code needs, so there is never a legitimate reason for that code to read from the
# filesystem itself (and the subprocess backend has no sandboxing that would stop a local read).
BANNED_ATTRIBUTE_PREFIXES = ("read_",)
URL_PREFIXES = ("http://", "https://", "ftp://", "s3://", "gs://")


@dataclass
class GateResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def _is_banned_import_name(name: str) -> bool:
    """A name that must never be pulled in by `from <module> import <name>`: a banned reader/writer,
    a buffered-render method, or a read_* loader. (Private names and star imports are handled separately.)"""
    return (name in BANNED_ATTRIBUTES or name in BANNED_WRITER_ATTRIBUTES
            or name in BUFFERED_RENDER_ATTRIBUTES or name.startswith(BANNED_ATTRIBUTE_PREFIXES))


def _json_bound_names(tree: ast.AST) -> set[str]:
    """Names bound to the json module by `import json` / `import json as <alias>`.
    json.dump/json.dumps are allowed only on such a name. `from json import ...` is not tracked here --
    it does not bind the module, and its imported names are screened by the import rules like any other."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "json":
                    names.add(alias.asname or "json")
    return names


def check_code(code: str) -> GateResult:
    """Static screen for careless or obviously hostile code. Not a security boundary; the runner is."""
    reasons: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return GateResult(False, [f"syntax error: {e.msg} (line {e.lineno})"])
    json_names = _json_bound_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # Only a bare top-level allowed module -- "import numpy", "import numpy as np".
            # A dotted import ("import numpy.lib._datasource") reaches submodules and is rejected, because a
            # submodule can expose readers/writers (and openers) the attribute rules below never see.
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    reasons.append(f"import of '{alias.name}' is not allowed; allowed (top-level only): {sorted(ALLOWED_IMPORTS)}")
        elif isinstance(node, ast.ImportFrom):
            # The source must be a top-level allowed module (no dots, no relative import),
            # and each imported name must not be a banned reader/writer, a private name, or a star import.
            module = node.module
            if node.level or module is None or module not in ALLOWED_IMPORTS:
                reasons.append(f"import from '{module}' is not allowed; allowed (top-level only): {sorted(ALLOWED_IMPORTS)}")
            else:
                for alias in node.names:
                    if alias.name == "*":
                        reasons.append(f"star import from '{module}' is not allowed")
                    elif alias.name.startswith("_"):
                        reasons.append(f"import of private name '{alias.name}' from '{module}' is not allowed")
                    elif _is_banned_import_name(alias.name):
                        reasons.append(f"import of '{alias.name}' from '{module}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                reasons.append(f"dunder attribute access '{node.attr}' is not allowed")
            if node.attr in JSON_SERIALISER_ATTRIBUTES:
                # Allowed only on a name bound to the json module; banned on every other receiver.
                if not (isinstance(node.value, ast.Name) and node.value.id in json_names):
                    reasons.append(f"banned attribute '{node.attr}'")
            elif (node.attr in BANNED_ATTRIBUTES or node.attr in BANNED_WRITER_ATTRIBUTES
                    or node.attr.startswith(BANNED_ATTRIBUTE_PREFIXES)):
                reasons.append(f"banned attribute '{node.attr}'")
        elif isinstance(node, ast.Name):
            if node.id in BANNED_CALLS:
                reasons.append(f"banned call '{node.id}'")
            if node.id.startswith("__"):
                reasons.append(f"dunder name '{node.id}' is not allowed; closes access to __builtins__ and similar")
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str) and "__" in node.value:
                reasons.append(f"string constant {node.value!r} containing '__' is not allowed "
                               "(dunder); closes format-string traversal such as '{0.__class__}'.format(x)")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr.startswith("read_"):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith(URL_PREFIXES):
                        reasons.append(f"url argument to {attr} is not allowed; data is preloaded")
            if attr in BUFFERED_RENDER_ATTRIBUTES:
                # A positional argument or a buf= keyword is the file/buffer sink; the no-argument
                # form returns a string and is allowed. A *args unpack counts as positional.
                if node.args or any(kw.arg == "buf" for kw in node.keywords):
                    reasons.append(f"banned file/buffer argument to {attr}; the no-argument form returns a string")
    return GateResult(not reasons, reasons)

import pytest

@pytest.mark.parametrize("code", [
    "import pandas as pd\nresult = {}",
    "from statistics import mean\nresult = {'a': mean([1, 2])}",
    "import numpy as np\nimport math\nresult = {'v': float(np.sqrt(4)) + math.pi}",
    "x = df['close'].to_numpy()\nresult = {}",
    "y = df['close'].to_list()\nresult = {}",
    "z = df.to_dict()\nresult = {}",
    # B5: the no-argument render forms return strings and stay allowed.
    "s = df.to_string()\nresult = {'s': s}",
    "h = df.to_html()\nresult = {'h': h}",
    "m = df.to_markdown()\nresult = {'m': m}",
    # Final residual 3: json.dump/json.dumps are allowed on the json module (json.dumps returns a string).
    "import json\ns = json.dumps({'a': 1})\nresult = {}",
    "import json as j\ns = j.dumps({'a': 1})\nresult = {}",
    # Final residual 1a: a top-level from-import of a non-banned name from an allowed module stays allowed.
    "from numpy import array\nx = array([1, 2])\nresult = {}",
    "from datetime import datetime, timedelta\nresult = {}",
    "from statistics import mean, stdev\nresult = {}",
])
def test_allowed_code_passes(code):
    from nasdaq_agent.sandbox.gate import check_code
    assert check_code(code).ok

@pytest.mark.parametrize("code, reason", [
    ("import os\nresult = {}", "import"),
    ("import socket", "import"),
    ("from urllib import request", "import"),
    ("x = df.__class__", "dunder"),
    ("open('/etc/passwd')", "banned call"),
    ("eval('1')", "banned call"),
    ("getattr(df, 'to_csv')", "banned call"),
    ("df.query('@__builtins__')", "banned attribute"),
    ("pd.read_pickle('x')", "banned attribute"),
    ("pd.read_csv('http://evil.example/x.csv')", "url"),
    ("__import__('os')", "banned call"),
    ("def f(:\n pass", "syntax"),
    ("x = __builtins__", "dunder"),
    ("s = '{0.__class__}'.format(1)", "dunder"),
    ("df2 = pd.read_csv('local.csv')", "banned attribute"),
    ("a = np.loadtxt('x.txt')", "banned attribute"),
    ("df.to_csv('/tmp/x.csv')", "banned attribute"),
    ("np.savetxt('x.txt', df.values)", "banned attribute"),
    ("np.DataSource().open('/etc/hosts')", "banned attribute"),
    ("pd.ExcelFile('a.xlsx')", "banned attribute"),
    ("p = pd.io.parsers", "banned attribute"),
    # B5: readers/writers reachable by attribute name.
    ("rows = np.fromregex('../../../.env', r'(.+)', dtype=[('l', 'U200')])", "fromregex"),
    ("df['close'].to_numpy().dump('../../../x.pkl')", "dump"),
    ("b = np.array([1]).dumps()", "dumps"),
    # B5: the render family writes a file/buffer when given a positional arg or a buf= keyword.
    ("df.to_string('../../../x.txt')", "to_string"),
    ("df.to_string(buf='../../../x.txt')", "to_string"),
    ("df.to_html('../../../x.html')", "to_html"),
    ("df.to_latex(buf='../../../x.tex')", "to_latex"),
    ("df.to_markdown('../../../x.md')", "to_markdown"),
    # Final residual 1a: only top-level module imports; no dotted/submodule imports, no reader/writer or
    # private from-imports, no star imports.
    ("import numpy.lib._datasource as ds\nds.open('x')", "not allowed"),
    ("from numpy.lib import npyio", "not allowed"),
    ("from numpy import fromregex", "fromregex"),
    ("from numpy import loadtxt", "loadtxt"),
    ("from numpy import genfromtxt", "genfromtxt"),
    ("from numpy import save", "save"),
    ("from numpy import fromfile", "fromfile"),
    ("from pandas import read_csv", "read_csv"),
    ("from numpy import *", "star import"),
    ("from numpy import _core", "private name"),
    ("from . import helpers", "not allowed"),
    # Final residual 3: dump/dumps are still banned on any non-json receiver.
    ("arr = df['close'].to_numpy()\narr.dump('../../../x.pkl')", "dump"),
    ("import numpy as np\nb = np.array([1]).dumps()", "dumps"),
])
def test_rejected_code(code, reason):
    from nasdaq_agent.sandbox.gate import check_code
    res = check_code(code)
    assert res.ok is False and any(reason in r for r in res.reasons), res.reasons

"""The in-app Python REPL executes arbitrary code, so its gate is tested end to end:
through the real Dash HTTP endpoint (/_dash-update-component), from loopback and from
a remote address, in each of the three import-time modes of XENIUM_ENABLE_REPL.

The mode is read once at import, so the two enabled modes each run in a fresh Python
process on a private copy of the tiny bundle (tl.run_in_fresh_app).
"""
from __future__ import annotations

import json

import os

import pytest

import xe_testlib as tl

# Runs inside the app process. The REPL code plants a marker in DATA; whether it appears
# tells whether the code ran.
PROBE = r'''
import time, threading

def fire(code, addr, headers=None):
    client = xe.app.server.test_client()
    body = {"output": "repl-input.value",
            "outputs": {"id": "repl-input", "property": "value"},
            "inputs": [{"id": "repl-run", "property": "n_clicks", "value": 1},
                       {"id": "repl-input", "property": "n_submit", "value": None}],
            "changedPropIds": ["repl-run.n_clicks"],
            "state": [{"id": "repl-input", "property": "value", "value": code}]}
    before = set(threading.enumerate())
    r = client.post("/_dash-update-component", json=body, environ_base={"REMOTE_ADDR": addr},
                    headers=headers or {})
    for t in set(threading.enumerate()) - before:
        t.join(20)
    return r.status_code

def attempt(name, addr, headers=None):
    xe.DATA.pop("_repl_marker", None)
    status = fire("DATA.__setitem__('_repl_marker', %r)" % name, addr, headers)
    ran = xe.DATA.pop("_repl_marker", None) == name
    return {"status": status, "ran": ran}

RESULT["mode"] = [xe._REPL_ENABLED, xe._REPL_ALLOW_REMOTE]
RESULT["loopback4"] = attempt("a", "127.0.0.1")
RESULT["loopback6"] = attempt("b", "::1")
RESULT["remote"] = attempt("c", "10.20.30.40")
RESULT["remote_spoofing_xff"] = attempt("d", "10.20.30.40", {"X-Forwarded-For": "127.0.0.1",
                                                              "X-Real-IP": "127.0.0.1",
                                                              "Forwarded": "for=127.0.0.1"})
RESULT["remote_public"] = attempt("e", "203.0.113.9")
RESULT["log"] = [l for l in xe._log_buffer if "REPL" in l]
import plotly
RESULT["layout_says_disabled"] = "REPL disabled" in json.dumps(xe.app.layout, cls=plotly.utils.PlotlyJSONEncoder)
'''


def _probe_in_process(xe):
    ns = {"xe": xe, "RESULT": {}, "json": json}
    exec(PROBE, ns)
    return ns["RESULT"]


def test_repl_is_off_by_default(xe, capsys):
    out = _probe_in_process(xe)
    assert "REPL is disabled" in capsys.readouterr().out
    assert out["mode"] == [False, False]
    for who in ("loopback4", "loopback6", "remote", "remote_spoofing_xff", "remote_public"):
        assert out[who]["ran"] is False, f"REPL code ran for {who} although the REPL is disabled"
        assert out[who]["status"] == 200, "the callback answers (and clears the input) without executing"
    assert out["layout_says_disabled"]


def test_repl_callable_directly_still_refuses_when_disabled(xe):
    """Belt and braces: the function itself, not only the HTTP layer, holds the gate."""
    with xe.app.server.test_request_context("/", environ_base={"REMOTE_ADDR": "127.0.0.1"}):
        with tl.new_threads():
            assert xe.run_repl(1, None, "DATA.__setitem__('_repl_marker', 1)") == ""
    assert "_repl_marker" not in xe.DATA
    with tl.new_threads():
        assert xe.run_repl(1, None, "   ") == ""


@pytest.fixture(scope="module")
def enabled_runs(tmp_path_factory):
    """The probe run once per enabled mode, the two app processes side by side."""
    from concurrent.futures import ThreadPoolExecutor
    if not os.path.exists(os.path.join(str(tl.TINY_BUNDLE), "experiment.xenium")):
        pytest.skip(f"no Xenium bundle at {tl.TINY_BUNDLE}")      # as the xe fixture does
    before = tl.bundle_snapshot(tl.TINY_BUNDLE)

    def run(mode):
        root = tmp_path_factory.mktemp(f"repl_{mode}")
        (root / "cache").mkdir()
        bundle = tl.make_working_copy(tl.TINY_BUNDLE, root / "tiny")
        return tl.run_in_fresh_app(PROBE, bundle, root / "cache", env={"XENIUM_ENABLE_REPL": mode})

    with ThreadPoolExecutor(2) as pool:
        out = dict(zip(("1", "remote"), pool.map(run, ("1", "remote"))))
    assert tl.bundle_snapshot(tl.TINY_BUNDLE) == before, "tests/data was modified"
    return out


def test_repl_enabled_is_loopback_only(enabled_runs):
    out = enabled_runs["1"]
    assert out["mode"] == [True, False]
    assert out["loopback4"]["ran"] and out["loopback6"]["ran"], "local requests must work when enabled"
    for who in ("remote", "remote_spoofing_xff", "remote_public"):
        assert out[who]["ran"] is False, f"REPL executed code for {who}"
    assert any("not from localhost" in line for line in out["log"])
    assert not out["layout_says_disabled"]


def test_repl_remote_mode_is_explicit_opt_in(enabled_runs):
    out = enabled_runs["remote"]
    assert out["mode"] == [True, True]
    assert all(out[who]["ran"] for who in ("loopback4", "remote", "remote_public"))


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "2", "enable", "REMOTELY"])
def test_repl_mode_parsing_fails_closed(value):
    """Only the documented values enable the REPL; anything else leaves it off. The parsing
    lines are taken from the app source so that this does not need a process per value."""
    import ast
    tree = tl.app_tree()
    wanted = ("_REPL_MODE", "_REPL_ENABLED", "_REPL_ALLOW_REMOTE")
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets)]
    assert len(nodes) == 3
    fake = type("E", (), {"environ": {"XENIUM_ENABLE_REPL": value}})
    ns = {"os": fake}
    exec(tl.compile_block(nodes), ns)
    assert ns["_REPL_ENABLED"] is False and ns["_REPL_ALLOW_REMOTE"] is False
    for on, remote in (("1", False), ("TRUE", False), (" yes ", False), ("on", False), ("Remote", True)):
        ns = {"os": type("E", (), {"environ": {"XENIUM_ENABLE_REPL": on}})}
        exec(tl.compile_block(nodes), ns)
        assert ns["_REPL_ENABLED"] is True and ns["_REPL_ALLOW_REMOTE"] is remote


def test_server_binds_loopback_by_default():
    """`python xenium_explorer.py` must not listen on all interfaces unless told to. The
    host expression of the __main__ block is evaluated with an empty environment."""
    import ast
    tree = tl.app_tree()
    main = [n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)]
    assert len(main) == 1
    hosts = [n for n in ast.walk(main[0]) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "_host" for t in n.targets)]
    assert len(hosts) == 1, "expected one `_host = ...` in the __main__ block"
    run = [n for n in ast.walk(main[0]) if isinstance(n, ast.Call) and ast.unparse(n.func) == "app.run"]
    assert run and any(k.arg == "host" and ast.unparse(k.value) == "_host" for k in run[0].keywords)
    assert not any(k.arg == "debug" and getattr(k.value, "value", None) is True for k in run[0].keywords), \
        "the Werkzeug debugger must stay off"

    def host_for(environ):
        ns = {"os": type("E", (), {"environ": environ})}
        exec(tl.compile_block(hosts), ns)
        return ns["_host"]

    assert host_for({}) in ("127.0.0.1", "localhost", "::1")
    assert host_for({"PORT": "9000", "XENIUM_ENABLE_REPL": "remote"}) in ("127.0.0.1", "localhost", "::1")

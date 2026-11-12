"""Static (AST) and runtime audit of the Dash callbacks.

The static half needs no import of the app, so it stays fast and also works when the
app cannot load a dataset. It catches the mistakes Dash reports only in the browser:
a callback whose parameter count differs from its Input/State count, a return tuple
of the wrong length, two callbacks writing one output without allow_duplicate, and a
dependency on a component id that no layout code creates.
"""
from __future__ import annotations

import ast
import collections
import json
import re

import pytest

import xe_testlib as tl


def _name(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _flatten(node):
    if isinstance(node, (ast.List, ast.Tuple)):
        for e in node.elts:
            yield from _flatten(e)
    else:
        yield node


def _dep(node):
    """(kind, component id, property, allow_duplicate) for an Input/State/Output call."""
    if not (isinstance(node, ast.Call) and _name(node.func) in ("Input", "State", "Output")):
        return None
    a = node.args
    cid = a[0].value if a and isinstance(a[0], ast.Constant) else ("<dyn>" if a else None)
    if a and isinstance(a[0], ast.Dict):
        cid = "<pattern>"
    prop = a[1].value if len(a) > 1 and isinstance(a[1], ast.Constant) else None
    for k in node.keywords:
        if k.arg == "component_id" and isinstance(k.value, ast.Constant):
            cid = k.value.value
        if k.arg == "component_property" and isinstance(k.value, ast.Constant):
            prop = k.value.value
    dup = any(k.arg == "allow_duplicate" and getattr(k.value, "value", False) for k in node.keywords)
    return _name(node.func), cid, prop, dup


class Audit:
    def __init__(self, tree):
        self.ids = set()
        self.id_patterns = []
        for n in ast.walk(tree):
            if isinstance(n, ast.keyword) and n.arg == "id":
                if isinstance(n.value, ast.Constant):
                    self.ids.add(n.value.value)
                elif isinstance(n.value, ast.JoinedStr):   # id=f"{prefix}-width" helpers
                    pat = "".join(re.escape(v.value) if isinstance(v, ast.Constant) else ".+"
                                  for v in n.value.values)
                    self.id_patterns.append(re.compile("^" + pat + "$"))
        self.callbacks = []          # (function node, [deps])
        self.clientside = []         # (lineno, [deps])
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef):
                for d in fn.decorator_list:
                    if isinstance(d, ast.Call) and _name(d.func) == "callback":
                        deps = [x for a in d.args for x in map(_dep, _flatten(a)) if x]
                        for k in d.keywords:
                            if k.arg in ("inputs", "state", "output"):
                                deps += [x for x in map(_dep, _flatten(k.value)) if x]
                        self.callbacks.append((fn, deps))
            if isinstance(fn, ast.Call) and _name(fn.func) == "clientside_callback":
                deps = [x for a in fn.args[1:] for x in map(_dep, _flatten(a)) if x]
                self.clientside.append((fn.lineno, deps))

    def known(self, cid) -> bool:
        return (cid in ("<dyn>", "<pattern>", None) or cid in self.ids
                or any(p.match(cid) for p in self.id_patterns))


@pytest.fixture(scope="module")
def audit():
    return Audit(tl.app_tree())


def test_audit_sees_the_callbacks(audit):
    # guards the audit itself: a refactor that renames the decorator must not turn
    # every check below into a vacuous pass
    assert len(audit.callbacks) >= 80
    assert len(audit.clientside) >= 10
    assert len(audit.ids) >= 250


def test_parameter_count_matches_inputs_and_states(audit):
    bad = []
    for fn, deps in audit.callbacks:
        n_in = sum(1 for d in deps if d[0] in ("Input", "State"))
        a = fn.args
        n_params = len(a.args) + len(a.posonlyargs)
        n_required = n_params - len(a.defaults)
        if not a.vararg and not (n_required <= n_in <= n_params):
            bad.append(f"{fn.name} (line {fn.lineno}): {n_in} Input/State vs {n_params} parameters")
    assert not bad, "\n".join(bad)


def test_return_tuples_match_output_count(audit):
    bad = []
    for fn, deps in audit.callbacks:
        n_out = sum(1 for d in deps if d[0] == "Output")
        if n_out < 2:
            continue
        # returns of nested helper functions are not the callback's own
        nested = {id(r) for sub in ast.walk(fn) if isinstance(sub, (ast.FunctionDef, ast.Lambda))
                  and sub is not fn for r in ast.walk(sub)}
        for r in ast.walk(fn):
            if (isinstance(r, ast.Return) and id(r) not in nested and isinstance(r.value, ast.Tuple)
                    and not any(isinstance(e, ast.Starred) for e in r.value.elts)
                    and len(r.value.elts) != n_out):
                bad.append(f"{fn.name} (line {r.lineno}): returns {len(r.value.elts)} values for {n_out} outputs")
    assert not bad, "\n".join(bad)


def test_every_dependency_id_is_created_somewhere(audit):
    bad = []
    for fn, deps in audit.callbacks:
        bad += [f"{fn.name} (line {fn.lineno}): {k}({cid!r}, {prop!r})"
                for k, cid, prop, _ in deps if not audit.known(cid)]
    for lineno, deps in audit.clientside:
        bad += [f"clientside_callback (line {lineno}): {k}({cid!r}, {prop!r})"
                for k, cid, prop, _ in deps if not audit.known(cid)]
    assert not bad, "\n".join(bad)


def test_no_output_written_twice_without_allow_duplicate(audit):
    writers = collections.defaultdict(list)
    for fn, deps in audit.callbacks:
        for k, cid, prop, dup in deps:
            if k == "Output" and cid not in ("<dyn>", "<pattern>"):
                writers[(cid, prop)].append((fn.name, fn.lineno, dup))
    for lineno, deps in audit.clientside:
        for k, cid, prop, dup in deps:
            if k == "Output" and cid not in ("<dyn>", "<pattern>"):
                writers[(cid, prop)].append(("<clientside>", lineno, dup))
    bad = {key: w for key, w in writers.items() if sum(1 for x in w if not x[2]) > 1}
    assert not bad, json.dumps({str(k): v for k, v in bad.items()}, indent=1)


def test_clientside_functions_exist_in_assets(audit):
    """ClientsideFunction(namespace, function_name) must name a function the asset
    scripts register on window.dash_clientside."""
    tree = tl.app_tree()
    wanted = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and _name(n.func) == "ClientsideFunction":
            kw = {k.arg: k.value.value for k in n.keywords if isinstance(k.value, ast.Constant)}
            args = [a.value for a in n.args if isinstance(a, ast.Constant)]
            wanted.add((kw.get("namespace", args[0] if args else None),
                        kw.get("function_name", args[1] if len(args) > 1 else None)))
    assert wanted, "no ClientsideFunction found (audit out of date?)"
    js = "\n".join(p.read_text(errors="replace") for p in (tl.REPO / "assets").glob("*.js"))
    for namespace, function in wanted:
        assert re.search(rf"\b{re.escape(namespace)}\b\s*:", js), f"namespace {namespace} not registered in assets/*.js"
        assert re.search(rf"\b{re.escape(function)}\b\s*[:(=]", js), f"{namespace}.{function} not defined in assets/*.js"


# ── runtime half: the same questions asked of the registered app ─────────────
def _walk(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    for child in (children if isinstance(children, (list, tuple)) else [children]):
        if hasattr(child, "to_plotly_json"):
            yield from _walk(child)


def test_layout_ids_are_unique_and_layout_serialises(xe):
    import plotly
    layout = xe.app.layout() if callable(xe.app.layout) else xe.app.layout
    ids = [c.id for c in _walk(layout) if isinstance(getattr(c, "id", None), str)]
    dup = [i for i, n in collections.Counter(ids).items() if n > 1]
    assert not dup, f"duplicate component ids in the layout: {dup}"
    assert len(ids) > 200
    text = json.dumps(layout, cls=plotly.utils.PlotlyJSONEncoder)
    assert len(text) > 10_000


def test_registered_callbacks_agree_with_the_static_audit(xe, audit):
    """Dash registered as many server callbacks as the source declares, and each one's
    Input+State list fits its function's signature."""
    import inspect
    registered = {k: v for k, v in xe.app.callback_map.items() if v.get("callback") is not None}
    assert len(registered) == len(audit.callbacks)
    by_name = {fn.name: fn for fn, _ in audit.callbacks}
    assert len(by_name) == len(audit.callbacks), "two callbacks share a function name"
    for output_id, spec in registered.items():
        n_in = len(spec["inputs"]) + len(spec["state"])
        fn = spec["callback"]
        fn = getattr(fn, "__wrapped__", fn)
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        required = [p for p in params if p.default is p.empty]
        assert len(required) <= n_in <= len(params), (fn.__name__, output_id, n_in, len(params))


def test_static_layout_ids_cover_callback_inputs(xe, audit):
    """Inputs that are plain string ids must exist in the initial layout or be created by
    layout code elsewhere in the source (dynamic content), never be a typo."""
    layout = xe.app.layout() if callable(xe.app.layout) else xe.app.layout
    in_layout = {c.id for c in _walk(layout) if isinstance(getattr(c, "id", None), str)}
    missing = set()
    for spec in xe.app.callback_map.values():
        for dep in list(spec["inputs"]) + list(spec["state"]):
            cid = dep["id"]
            if isinstance(cid, str) and not cid.startswith("{") and cid not in in_layout and not audit.known(cid):
                missing.add(cid)
    assert not missing, sorted(missing)


# ── every Python source at least parses ──────────────────────────────────────
@pytest.mark.parametrize("path", sorted([*tl.app_sources(), tl.REPO / "xenium_viewer_api.py",
                                         *(tl.REPO / "functions").glob("*.py"),
                                         tl.REPO / "viewer_src" / "dev_server.py"]),
                         ids=lambda p: p.name)
def test_source_parses(path):
    """The standalone scripts are not imported by any other test (most need R or large
    inputs), so a syntax error in one of them would otherwise go unnoticed."""
    ast.parse(path.read_text(), filename=str(path))

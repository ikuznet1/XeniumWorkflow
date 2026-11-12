"""The app is a package (xenium_app) behind one namespace (xenium_explorer).

Tests, scratch scripts and the in-app REPL treat the app as one module whose globals can be
read AND rebound: ``xe.CACHE_BASE_DIR = ...``, ``monkeypatch.setattr(xe, "_f", stub)``.
That only works if a global is defined in exactly one module and every other module looks
it up there at the moment of use (``owner.NAME``). The first half checks the source for
that rule; the second half rebinds a constant, a state dict, a counter and a function
through ``xe`` and through the REPL namespace and watches the code that uses them.
"""
from __future__ import annotations

import ast
import builtins
import dis
import os
import sys
import types

import pytest

import xe_testlib as tl

PKG = tl.APP_PACKAGE.name
_own_vars = types.ModuleType.__dict__["__dict__"].__get__      # a module's own dict (vars(xe) is a view)
MODULE_FILES = [p for p in tl.app_sources() if p.parent == tl.APP_PACKAGE and p.name != "__init__.py"]
MODULE_NAMES = [p.stem for p in MODULE_FILES]


def _toplevel_names(tree) -> set:
    """Names a module defines itself: assignments, defs and classes at module level
    (also inside if / with / for / try), and names its functions declare ``global``."""
    out = set()

    def targets(node):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                out.add(sub.id)

    def block(stmts):
        for n in stmts:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(n.name)
                if not isinstance(n, ast.ClassDef):
                    out.update(nm for g in ast.walk(n) if isinstance(g, ast.Global) for nm in g.names)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    targets(t)
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
                targets(n.target)
            elif isinstance(n, (ast.For, ast.With)):
                for t in ([n.target] if isinstance(n, ast.For) else
                          [i.optional_vars for i in n.items if i.optional_vars is not None]):
                    targets(t)
                block(n.body)
                block(getattr(n, "orelse", []))
            elif isinstance(n, ast.If):
                block(n.body)
                block(n.orelse)
            elif isinstance(n, ast.Try):
                block(n.body)
                block(n.orelse)
                block(n.finalbody)
                for h in n.handlers:
                    block(h.body)
    block(tree.body)
    return out


# ── the rule, in the source ──────────────────────────────────────────────────
def test_every_global_is_defined_in_one_module():
    seen = {}
    for path in MODULE_FILES:
        for name in _toplevel_names(ast.parse(path.read_text())):
            seen.setdefault(name, []).append(path.stem)
    twice = {k: v for k, v in seen.items() if len(v) > 1}
    assert not twice, f"defined in more than one module (xe.<name> would be ambiguous): {twice}"


@pytest.mark.parametrize("path", MODULE_FILES, ids=lambda p: p.name)
def test_modules_never_copy_each_others_globals(path):
    """`from .core import CACHE_BASE_DIR` would freeze the value at import: a later
    `xe.CACHE_BASE_DIR = ...` would not reach this module."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").split(".")[0] == PKG):
            assert node.module in (None, PKG), \
                f"{path.name}:{node.lineno}: imports names from a sibling module; import the module"
            assert all(a.name in MODULE_NAMES and a.asname is None for a in node.names), \
                f"{path.name}:{node.lineno}: sibling modules are imported under their own name"
        if isinstance(node, ast.Import):
            assert not any(a.name.split(".")[0] == PKG for a in node.names), f"{path.name}:{node.lineno}"


def test_library_modules_do_not_import_the_startup_modules():
    """Importing startup / app_layout / cb_* / viewer_provider does the import-time work
    (data load, app, callbacks). Only the package __init__ decides when that happens."""
    init = ast.parse((tl.APP_PACKAGE / "__init__.py").read_text())
    order = next(ast.literal_eval(n.value) for n in init.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "MODULES" for t in n.targets))
    first = order.index("startup")
    libraries, work = order[:first], order[first:]
    assert set(order) | {"compat"} == set(MODULE_NAMES)

    def siblings(name):
        tree = ast.parse((tl.APP_PACKAGE / f"{name}.py").read_text())
        return {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level
                for a in n.names}

    assert siblings("core") == set() and siblings("compat") == set()
    for name in libraries:
        assert not siblings(name) & set(work), f"{name} imports {siblings(name) & set(work)}"
    for i, name in enumerate(work):          # ... and the startup modules only look back
        assert not siblings(name) & set(work[i + 1:]), f"{name} imports {siblings(name) & set(work[i + 1:])}"


def _code_objects(code):
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _code_objects(const)


@pytest.mark.parametrize("path", MODULE_FILES, ids=lambda p: p.name)
def test_every_global_reference_resolves(xe, path):
    """No name was left behind by the split: each global a function reads exists in its
    module (or is a builtin), and each ``module.NAME`` exists in that module."""
    mod = sys.modules[f"{PKG}.{path.stem}"]
    source = path.read_text()
    missing = set()
    for code in _code_objects(compile(source, str(path), "exec")):
        for ins in dis.get_instructions(code):
            if ins.opname in ("LOAD_GLOBAL", "STORE_GLOBAL", "DELETE_GLOBAL") \
                    and ins.argval not in mod.__dict__ and not hasattr(builtins, ins.argval):
                missing.add(ins.argval)
    assert not missing, f"{path.name}: undefined globals {sorted(missing)}"
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in MODULE_NAMES and isinstance(mod.__dict__.get(node.value.id), types.ModuleType):
            owner = sys.modules[f"{PKG}.{node.value.id}"]
            if not hasattr(owner, node.attr):
                missing.add(f"{node.value.id}.{node.attr} (line {node.lineno})")
    assert not missing, f"{path.name}: {sorted(missing)}"


def test_xe_exposes_the_globals_of_every_module(xe):
    for name in MODULE_NAMES:
        if name == "compat":
            continue
        mod = sys.modules[f"{PKG}.{name}"]
        for key in _toplevel_names(ast.parse((tl.APP_PACKAGE / f"{name}.py").read_text())):
            if key in mod.__dict__:
                assert getattr(xe, key) is mod.__dict__[key], f"xe.{key} is not {name}.{key}"
                assert key in dir(xe)
    with pytest.raises(AttributeError):
        xe._no_such_global_anywhere


# ── rebinding through xe reaches the code that uses the name ─────────────────
def test_rebinding_a_constant_reaches_other_modules(xe, monkeypatch, tmp_path):
    core = sys.modules[f"{PKG}.core"]
    before = xe.CACHE_BASE_DIR
    monkeypatch.setattr(xe, "CACHE_BASE_DIR", str(tmp_path))
    assert core.CACHE_BASE_DIR == str(tmp_path)
    assert xe._cache_dir() == str(tmp_path)                                    # caching.py
    assert os.path.dirname(xe._roi_cache_path()) == str(tmp_path)              # regions.py
    assert os.path.dirname(xe._sdata_cache_path()) == str(tmp_path)            # sdata_store.py
    assert os.path.dirname(xe._spage_index_path()) == str(tmp_path)            # spage.py
    monkeypatch.undo()
    assert xe.CACHE_BASE_DIR == before == core.CACHE_BASE_DIR and xe._cache_dir() == before
    xe.CACHE_BASE_DIR = str(tmp_path)              # plain assignment, as scripts and the REPL do
    try:
        assert xe._cache_dir() == str(tmp_path) == vars(xe)["CACHE_BASE_DIR"]
        assert "CACHE_BASE_DIR" not in _own_vars(xe), "the name must not be copied onto xenium_explorer"
    finally:
        xe.CACHE_BASE_DIR = before
    assert xe._cache_dir() == before


def test_rebinding_a_state_dict_reaches_other_modules(xe, monkeypatch):
    core = sys.modules[f"{PKG}.core"]
    assert xe._baysor_state is core._baysor_state
    assert not any(tool == "baysor" for tool, _ in xe._loaded_reseg_results())  # subsetting.py
    marker = {"cells_df": None}
    monkeypatch.setattr(xe, "_baysor_state", {"status": "done", "message": "", "result": marker})
    assert core._baysor_state["result"] is marker
    assert ("baysor", marker) in xe._loaded_reseg_results()
    monkeypatch.undo()
    assert not any(tool == "baysor" for tool, _ in xe._loaded_reseg_results())
    from unittest import mock
    with mock.patch.object(xe, "_baysor_state", {"status": "done", "message": "", "result": marker}):
        assert ("baysor", marker) in xe._loaded_reseg_results()
    assert xe._baysor_state is core._baysor_state and core._baysor_state.get("result") is None
    # a module that rebinds the dict itself (`global _spatial_state` when the app was one file)
    assert "_spatial_state" in vars(core) and "_spatial_state" not in vars(sys.modules[f"{PKG}.spatial"])


def test_rebinding_a_counter_reaches_its_users(xe):
    gen = xe._dataset_gen
    xe._check_dataset_gen(gen)                     # jobs.py reads its own global
    xe._dataset_gen = gen + 1
    try:
        assert sys.modules[f"{PKG}.jobs"]._dataset_gen == gen + 1
        with pytest.raises(xe._DatasetChanged):
            xe._check_dataset_gen(gen)
    finally:
        xe._dataset_gen = gen
    xe._check_dataset_gen(gen)


def test_rebinding_a_function_reaches_callers_in_and_outside_its_module(xe, monkeypatch):
    real = xe._roi_cache_path()
    real_run = xe._seg_run_dir("baysor", "deadbeef")
    monkeypatch.setattr(xe, "_dataset_key", lambda: "some other dataset")       # caching.py
    assert xe._dataset_key() == "some other dataset"
    assert xe._roi_cache_path() != real                                          # caller in regions.py
    assert xe._seg_run_dir("baysor", "deadbeef") != real_run                     # caller in caching.py
    monkeypatch.undo()
    assert xe._roi_cache_path() == real and xe._seg_run_dir("baysor", "deadbeef") == real_run


def test_new_attributes_stay_on_the_namespace_module(xe):
    xe._layout_test_marker = 1
    try:
        assert _own_vars(xe)["_layout_test_marker"] == 1 == vars(xe)["_layout_test_marker"]
        assert not any("_layout_test_marker" in vars(sys.modules[f"{PKG}.{n}"]) for n in MODULE_NAMES)
    finally:
        del xe._layout_test_marker
    assert not hasattr(xe, "_layout_test_marker")


# ── the REPL namespace ───────────────────────────────────────────────────────
def test_repl_namespace_reads_and_rebinds_app_globals(xe, tmp_path):
    ns = xe._repl_globals
    core = sys.modules[f"{PKG}.core"]
    for name in ("DATA", "subset", "unsubset", "run_spage", "get_genes", "retry_split_zarr_write",
                 "to_spatialdata", "create_patches", "segment_tissue_roi", "app", "np", "pd",
                 "_baysor_state", "CACHE_BASE_DIR", "make_spatial_fig"):
        assert eval(name, ns) is getattr(xe, name), name
        assert name in ns
    assert eval("len(DATA['df'])", ns) == len(xe.DATA["df"])
    assert eval("[k for k in ('df', 'nope') if k in DATA]", ns) == ["df"]     # nested scope
    before = xe.CACHE_BASE_DIR
    try:
        exec(f"CACHE_BASE_DIR = {str(tmp_path)!r}", ns)
        assert core.CACHE_BASE_DIR == str(tmp_path) == xe._cache_dir()
        assert not dict.__contains__(ns, "CACHE_BASE_DIR")
    finally:
        xe.CACHE_BASE_DIR = before
    exec("def _repl_probe():\n    return _cache_dir(), len(DATA['df'])\n_repl_value = _repl_probe()", ns)
    try:
        assert ns["_repl_value"] == (before, len(xe.DATA["df"]))
        assert dict.__contains__(ns, "_repl_probe") and not hasattr(xe, "_repl_probe")
    finally:
        del ns["_repl_probe"], ns["_repl_value"]
    with pytest.raises(NameError):
        eval("_no_such_global_anywhere", ns)
    assert eval("len", ns) is len

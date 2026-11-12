#!/usr/bin/env python3
"""
Xenium Explorer Clone
Interactive visualization for 10x Genomics Xenium spatial transcriptomics data.

Usage:
    python xenium_explorer.py [path/to/output-XETG...]
    python xenium_explorer.py          # auto-detects output-* directory

The code lives in the xenium_app package (see xenium_app/__init__.py for the layout).
This file is the entry point and, for ``import xenium_explorer as xe``, a compatibility
module: every global of the app is reachable as ``xe.<name>``, and assigning
``xe.<name> = value`` rebinds the name in the module that defines it (xenium_app/compat.py).
"""

import sys as _sys
import types as _types
from collections.abc import MutableMapping as _MutableMapping

import xenium_app                      # loads the dataset, builds the app, registers callbacks
from xenium_app import compat as _compat

_own_dict = _types.ModuleType.__dict__["__dict__"].__get__     # the module's real __dict__


class _AppGlobals(_MutableMapping):
    """``vars(xenium_explorer)``: a live view of the globals of every module of the app.
    (pytest's monkeypatch and unittest.mock look a name up here to decide how to undo.)"""

    def __init__(self, module):
        self._own = _own_dict(module)

    def __getitem__(self, name):
        if name in self._own:
            return self._own[name]
        return _compat.lookup(name)

    def __setitem__(self, name, value):
        if name in self._own or not _compat.assign(name, value):
            self._own[name] = value

    def __delitem__(self, name):
        if name in self._own:
            del self._own[name]
        elif not _compat.delete(name):
            raise KeyError(name)

    def __iter__(self):
        return iter(set(self._own) | _compat.all_names())

    def __len__(self):
        return len(set(self._own) | _compat.all_names())


class _AppNamespace(_types.ModuleType):
    """This module, with attribute access forwarded to the modules of xenium_app."""

    def __getattr__(self, name):       # only called when the name is not set on this module
        try:
            return _compat.lookup(name)
        except KeyError:
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}") from None

    def __setattr__(self, name, value):
        if name in _own_dict(self) or not _compat.assign(name, value):
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in _own_dict(self) or not _compat.delete(name):
            super().__delattr__(name)

    def __dir__(self):
        return sorted(_AppGlobals(self))

    @property
    def __dict__(self):
        return _AppGlobals(self)


_sys.modules[__name__].__class__ = _AppNamespace


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging
    import os
    import signal as _signal
    app = xenium_app.app_layout.app
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    _port = int(os.environ.get("PORT", 8050))
    # Bind to localhost by default; set XENIUM_HOST=0.0.0.0 to serve other machines
    # (e.g. in a container). The generic HOST variable is deliberately not read:
    # csh/tcsh and conda compiler activation export it for other purposes.
    _host = os.environ.get("XENIUM_HOST", "").strip() or "127.0.0.1"
    if os.environ.get("HOST") and not os.environ.get("XENIUM_HOST"):
        print(f"  Note: HOST={os.environ['HOST']!r} is ignored - set XENIUM_HOST to change "
              f"the bind address (e.g. XENIUM_HOST=0.0.0.0 in a container).", flush=True)
    print(f"\nXenium Explorer running at http://localhost:{_port} "
          f"(bound to {_host}:{_port})\n", flush=True)
    # Baysor / Proseg run in their own session (so Cancel can stop the whole process
    # tree) and no longer receive the terminal's SIGHUP / a `kill` of this process:
    # stop them here, then let the signal take its default course.
    def _stop_tools_and_exit(signum, _frame):
        xenium_app.jobs._jobs_kill_all_processes()
        _signal.signal(signum, _signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for _signame in ("SIGTERM", "SIGHUP"):
        if hasattr(_signal, _signame):
            _signal.signal(getattr(_signal, _signame), _stop_tools_and_exit)

    app.run(debug=False, host=_host, port=_port)

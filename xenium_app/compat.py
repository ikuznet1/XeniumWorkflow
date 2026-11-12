"""One namespace over the modules of the package.

The app used to be a single file, so tests, scratch scripts and the in-app REPL read and
REBIND its globals (``xe.CACHE_BASE_DIR = ...``, ``monkeypatch.setattr(xe, "_f", stub)``).
The rule that keeps this working after the split:

* every global name is defined in exactly one module (its owner);
* code in the owner uses the bare name, code in any other module uses ``owner.NAME``
  (never ``from .owner import NAME``), so a name is always looked up in its owner's
  namespace at the moment it is used;
* ``xenium_explorer`` (the compatibility module) and the REPL namespace forward attribute
  reads, writes and deletes to the owner through the functions below.

This module imports nothing from the package; the other modules are found in sys.modules.
"""
import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The file the app is started through; data files that ship with the app sit next to it.
APP_FILE = os.path.join(REPO_DIR, "xenium_explorer.py")


def entry_name() -> str:
    """Import name handed to Dash / Flask: the module the app is started through, as when
    the app was one file (``__main__`` for ``python xenium_explorer.py``)."""
    if "xenium_explorer" in sys.modules:
        return "xenium_explorer"
    main = sys.modules.get("__main__")
    if os.path.basename(getattr(main, "__file__", None) or "") == "xenium_explorer.py":
        return "__main__"
    return "xenium_explorer"


def modules() -> list:
    """The loaded modules of the package, in import order."""
    pkg = sys.modules[__package__]
    found = (sys.modules.get(f"{__package__}.{name}") for name in pkg.MODULES)
    return [m for m in found if m is not None]


def owners(name: str) -> list:
    """Modules whose namespace holds *name*: one for a name of the app, several for an
    imported library alias such as ``np``."""
    if name.startswith("__") and name.endswith("__"):
        return []
    return [m for m in modules() if name in m.__dict__]


def lookup(name: str):
    """Value of the app global *name*; KeyError when no module defines it."""
    for m in owners(name):
        return m.__dict__[name]
    raise KeyError(name)


def assign(name: str, value) -> bool:
    """Rebind *name* where it is defined. False when no module defines it."""
    found = owners(name)
    for m in found:
        setattr(m, name, value)
    return bool(found)


def delete(name: str) -> bool:
    found = owners(name)
    for m in found:
        delattr(m, name)
    return bool(found)


def all_names() -> set:
    return {k for m in modules() for k in m.__dict__ if not (k.startswith("__") and k.endswith("__"))}


class ReplNamespace(dict):
    """Globals of the in-app REPL: every global of the app by its bare name. Assigning to a
    name the app defines rebinds it in its owner module; other names stay in this dict."""

    def __init__(self):
        super().__init__(__name__=entry_name())

    def __missing__(self, name):
        return lookup(name)

    def __setitem__(self, name, value):
        if dict.__contains__(self, name) or not assign(name, value):
            dict.__setitem__(self, name, value)

    def __delitem__(self, name):
        if dict.__contains__(self, name):
            dict.__delitem__(self, name)
        elif not delete(name):
            raise KeyError(name)

    def __contains__(self, name):
        return dict.__contains__(self, name) or bool(owners(name))

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def __iter__(self):
        return iter(set(dict.keys(self)) | all_names())

    def keys(self):
        return list(self)

    def __len__(self):
        return len(set(dict.keys(self)) | all_names())

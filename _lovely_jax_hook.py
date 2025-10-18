"""
This file should never be imported or executed by the user.

It gets called by the site hook `_lovely_jax_hook.pth` which in turn gets
called automatically by python upon startup (https://docs.python.org/3/library/site.html)
if the package lovely-jax is installed.

If the LOVELY_JAX environment variable is set to a "truthy" value, import and monkey patch
jax if it is already imported or gets imported in the future by the user.
"""

import os
import sys
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
from types import ModuleType
from typing import Optional, Sequence

def _after_import_jax():
    try:
        # lovely_jax auto-patches on import if LOVELY_JAX env var is set.
        import lovely_jax
    except Exception as e:
        ERROR_MESSAGE = """\
    Error: lovely_jax.monkey_patch() failed with:

    {}

    If you uninstalled lovely_jax, you should delete any '_lovely_jax_hook.pth'
    file on your system and unset your 'LOVELY_JAX' environment variable.
    """
        print(ERROR_MESSAGE.format(e), file=sys.stderr)

class _WrappedJaxLoader(importlib.abc.Loader):

    def __init__(self, real_loader: importlib.abc.Loader):
        self._real_loader = real_loader

    def create_module(self, spec: importlib.machinery.ModuleSpec):
        if hasattr(self._real_loader, "create_module"):
            return self._real_loader.create_module(spec)
        return None

    def exec_module(self, module: ModuleType):
        self._real_loader.exec_module(module)
        _after_import_jax()

class _JaxFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Optional[Sequence[str]], target: Optional[ModuleType] = None) -> Optional[importlib.machinery.ModuleSpec]:
        if fullname != 'jax':
            return None
        sys.meta_path.remove(self)
        real_spec = importlib.util.find_spec(fullname)
        if real_spec is None or real_spec.loader is None:
            return real_spec
        real_spec.loader = _WrappedJaxLoader(real_spec.loader)
        return real_spec

if os.environ.get("LOVELY_JAX", "").strip().lower() in {"1", "true", "yes"}:
    sys.meta_path.insert(0, _JaxFinder())

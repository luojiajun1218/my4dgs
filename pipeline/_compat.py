"""Keep libgs names and historical imports bound to the canonical modules.

No compatibility directory is needed. In particular, an old import must not
execute the same model file again under a second name, which would create
different dataclass/class identities and break pickle and isinstance checks.
"""

from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import find_spec, spec_from_loader
import sys


ALIASES = ('pipeline.ReCon-GS', 'pipeline.my4dgs')


class _AliasLoader(Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        module = import_module(self.target)
        self.canonical_spec = module.__spec__
        return module

    def exec_module(self, module):
        # importlib temporarily assigns the alias spec to the reused module.
        # Keep its canonical identity for inspect, reload, and new pickles.
        module.__spec__ = self.canonical_spec


class _AliasFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for prefix in ALIASES:
            if fullname.startswith(prefix + '.'):
                canonical = 'pipeline' + fullname[len(prefix):]
                spec = find_spec(canonical)
                if spec is None:
                    return None
                return spec_from_loader(fullname, _AliasLoader(canonical),
                    is_package=spec.submodule_search_locations is not None)
        return None


def install_aliases(package):
    for name in ALIASES:
        sys.modules[name] = package
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())

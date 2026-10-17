"""Project pipeline implementation with lazy libgs entry-point exports."""

from importlib import import_module
import sys

from ._compat import install_aliases


install_aliases(sys.modules[__name__])

__all__ = ['Config', 'Pipeline', 'DataConfig', 'DataModule', 'Module',
           'ModuleConfig', 'Trainer', 'TrainerConfig']


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    entry = import_module('.entry', __name__)
    globals().update({key: getattr(entry, key) for key in __all__})
    return globals()[name]

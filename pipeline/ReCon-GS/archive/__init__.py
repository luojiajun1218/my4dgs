"""Archived motion prototypes and historical diagnostics (read-only use)."""


def __getattr__(name):
    if name in {'Config', 'Pipeline', 'DataConfig', 'DataModule',
                'Module', 'ModuleConfig', 'Trainer', 'TrainerConfig'}:
        from . import legacy_runtime
        return getattr(legacy_runtime, name)
    raise AttributeError(name)

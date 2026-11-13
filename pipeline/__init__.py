"""pipeline 包的公开入口。

按需导出 entry.py 中的配置、管线、数据模块、模型和训练器，供 libgs 加载；
具体实现分别放在对应文件中，避免在包初始化时加载整个训练流程。
"""

from importlib import import_module

__all__ = ['Config', 'Pipeline', 'DataConfig', 'DataModule', 'Module',
           'ModuleConfig', 'Trainer', 'TrainerConfig']


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    entry = import_module('.entry', __name__)
    globals().update({key: getattr(entry, key) for key in __all__})
    return globals()[name]

import os
import importlib
import pkgutil

# 手动导入 config 和 registry，因为我们仍然需要它们
from .config import *
from .registry import registry

# 自动导入当前目录下所有的 .py 文件，但排除 config.py 和 registry.py
excluded_modules = ['config', 'registry', '__init__']
current_dir = os.path.dirname(__file__)

for _, module_name, _ in pkgutil.iter_modules([current_dir]):
    if module_name not in excluded_modules:
        module = importlib.import_module(f'.{module_name}', package=__name__)
        # 导入模块中的所有内容
        if hasattr(module, '__all__'):
            all_names = module.__all__
        else:
            all_names = [name for name in dir(module) if not name.startswith('_')]

        for name in all_names:
            globals()[name] = getattr(module, name)

__all__ = [
    'PLUGINS',
    'function_call_list',
    'get_tools_result_async',
] + list(registry.tools.keys())
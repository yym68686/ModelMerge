import os
import pkgutil
import importlib

# 首先导入registry，因为其他模块中的装饰器依赖它
from .registry import registry, register_tool, register_agent

# 自动导入当前目录下所有的插件模块
excluded_modules = ['config', 'registry', '__init__']
current_dir = os.path.dirname(__file__)

# 先导入所有模块，确保装饰器被执行
for _, module_name, _ in pkgutil.iter_modules([current_dir]):
    if module_name not in excluded_modules:
        importlib.import_module(f'.{module_name}', package=__name__)

# 然后从config导入必要的定义
from .config import *

# 确保将所有工具函数添加到全局名称空间
for tool_name, tool_func in registry.tools.items():
    globals()[tool_name] = tool_func

__all__ = [
    'PLUGINS',
    'function_call_list',
    'get_tools_result_async',
    'registry',
    'register_tool',
    'register_agent',
    'update_tools_config',
] + list(registry.tools.keys())
from pathlib import Path

from setuptools import setup, find_packages

setup(
    name="ModelMerge",
    version="0.1.5",
    description="ModelMerge is a multi-large language model API aggregator.",
    long_description=Path.open(Path("README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where=["models", "tools", "utils"]),
    package_dir={"": "."},
    install_requires=Path.open(Path("requirements.txt"), encoding="utf-8").read().splitlines(),
    # py_modules=["LLM-Hub"]
)
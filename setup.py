from pathlib import Path

from setuptools import setup, find_packages

setup(
    name="LLM-Hub",
    version="0.1.0",
    description="LLM-Hub is a multi-large language model API aggregator.",
    long_description=Path.open(Path("README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages("models"),
    package_dir={"": "src"},
    # py_modules=["LLM-Hub"]
)
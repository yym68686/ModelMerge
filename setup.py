from pathlib import Path

from setuptools import setup, find_packages

setup(
    name="modelmerge",
    version="1.0.9",
    description="modelmerge is a multi-large language model API aggregator.",
    long_description=Path.open(Path("README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=Path.open(Path("requirements.txt"), encoding="utf-8").read().splitlines(),
)
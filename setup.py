from pathlib import Path

from setuptools import setup, find_packages

setup(
    name="aient",
    version="1.0.38",
    description="Aient: The Awakening of Agent.",
    long_description=Path.open(Path("README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=Path.open(Path("requirements.txt"), encoding="utf-8").read().splitlines(),
    include_package_data=True,
)
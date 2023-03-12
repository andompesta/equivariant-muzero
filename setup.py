"""
This file configures the Python package with entrypoints used for future runs on Databricks.
Please follow the `entry_points` documentation for more details on how to configure the entrypoint:
* https://setuptools.pypa.io/en/latest/userguide/entry_point.html
"""

from setuptools import find_packages, setup
from semantic_retrieval import __version__

PACKAGE_REQUIREMENTS = [
    "wandb",
    "procgen",
    "ray",
]

DEV_REQUIREMENTS = [
    "",
]

setup(
    name="semantic_retrieval",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    install_requires=PACKAGE_REQUIREMENTS,
    extras_require={"dev": DEV_REQUIREMENTS},
    version=__version__,
    description="",
    author="",
)
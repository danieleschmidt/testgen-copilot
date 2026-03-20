from setuptools import setup, find_packages

setup(
    name="testgen-copilot",
    version="1.0.0",
    packages=find_packages(),
    entry_points={"console_scripts": ["testgen=testgen.cli:main"]},
)

import os

from setuptools import find_packages, setup

# Read requirements
requirements_path = "requirements.txt"
install_requires = []
if os.path.exists(requirements_path):
    with open(requirements_path, "r") as f:
        install_requires = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

# Read README for long description
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="qf",
    version="1.0.0",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.8",
)

"""Setup script for MARL Resource Allocation framework."""

from setuptools import setup, find_packages

setup(
    name="marl-resource-allocation",
    version="0.1.0",
    description=(
        "Multi-Agent Reinforcement Learning framework for adaptive "
        "resource allocation and task scheduling in heterogeneous "
        "distributed/cloud environments."
    ),
    author="Capstone Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "gymnasium>=0.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "tensorboard>=2.13.0",
        ],
    },
)

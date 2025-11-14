"""Setup configuration for RL-Control-Suite."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rl-control-suite",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-ready RL and optimal control library with safety guarantees",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rl-control-suite",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "scipy>=1.7.0",
        "gymnasium>=0.28.0",
        "matplotlib>=3.5.0",
        "tensorboard>=2.10.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinx-autodoc-typehints>=1.22.0",
        ],
        "mpc": [
            "casadi>=3.6.0",
        ],
        "jax": [
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
        ],
    },
)

from setuptools import setup, find_packages

setup(
    name="sde_solver",
    version="0.1.0",
    description="Vectorized Stochastic Differential Equation Solver using Rössler methods",
    author="Uri Maayan",
    author_email="uriuriuri7@hotmail.com",
    url="https://github.com/urimaayan/sde-solver",  # Update with actual repository
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "numba>=0.53.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "typing-extensions>=4.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.7",
)
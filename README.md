# SDE Solver

A high-performance library for solving Stochastic Differential Equations (SDEs) using vectorized operations and numerical methods based on stochastic Runge-Kutta schemes.

## Table of Contents

- [Introduction](#introduction)
- [Mathematical Background](#mathematical-background)
- [Features](#features)
- [Installation](#installation)
- [Quick Start Guide](#quick-start-guide)
- [Usage Examples](#usage-examples)
- [Example Implementations](#example-implementations)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Performance Considerations](#performance-considerations)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Introduction

This library provides efficient implementations of numerical methods for solving Itô stochastic differential equations with scalar noise. It utilizes advanced vectorized operations for performance and supports various simulation scenarios from simple test cases to complex real-world models.

The implementation focuses on:
- Numerical accuracy and stability
- Computational efficiency
- Flexible configuration and customization
- Comprehensive profiling and analysis

## Mathematical Background

### Stochastic Differential Equations

The library addresses Itô stochastic differential equations of the form:

$$dX_t = a(t, X_t) dt + b(t, X_t) dW_t$$

where:
- $X_t$ is the state vector at time $t$
- $a(t, X_t)$ is the drift coefficient (deterministic part)
- $b(t, X_t)$ is the diffusion coefficient (stochastic part)
- $W_t$ is a one-dimensional Wiener process (scalar noise)

### Stochastic Runge-Kutta Methods

The core of this library implements Stochastic Runge-Kutta (SRK) methods for the weak approximation of SDEs, following the approach developed by Rössler (2006). The general form of the SRK scheme for a step from $t_n$ to $t_{n+1}$ is:

$$Y_{n+1} = Y_n + \sum_{i=1}^{s} \alpha_i a(t_i^{(0)}, H_i^{(0)}) h_n + \sum_{i=1}^{s} \gamma_i^{(1)} b(t_i^{(1)}, H_i^{(1)}) I_{(1)} + \sum_{i=1}^{s} \gamma_i^{(2)} b(t_i^{(1)}, H_i^{(1)}) \frac{I_{(1,1)}}{\sqrt{h_n}}$$

where $H_i^{(k)}$ are the stage values defined by:

$$H_i^{(0)} = Y_n + \sum_{j=1}^{s} A_{ij}^{(0)} a(t_j^{(0)}, H_j^{(0)}) h_n + \sum_{j=1}^{s} B_{ij}^{(1)(0)} b(t_j^{(1)}, H_j^{(1)}) I_{(1)}$$

$$H_i^{(1)} = Y_n + \sum_{j=1}^{s} A_{ij}^{(1)} a(t_j^{(0)}, H_j^{(0)}) h_n + \sum_{j=1}^{s} B_{ij}^{(3)(1)} b(t_j^{(1)}, H_j^{(1)}) \sqrt{h_n}$$

and:
- $s$ is the number of stages
- $h_n$ is the step size
- $\alpha_i$, $\gamma_i^{(1)}$, $\gamma_i^{(2)}$, $A_{ij}^{(0)}$, $A_{ij}^{(1)}$, $B_{ij}^{(1)(0)}$, $B_{ij}^{(3)(1)}$ are the method coefficients
- $I_{(1)}$ and $I_{(1,1)}$ are random variables related to the Wiener process

The library implements several specific sets of coefficients that achieve different orders of convergence.

## Features

- **Vectorized Operations**: Efficient implementation using NumPy's vectorized operations
- **Adaptive Solvers**: Step size control for balancing accuracy and performance
- **Multiple SRK Schemes**: Support for various SRK coefficient sets including order 2.0 and 3.0 schemes
- **Flexible Configuration**: Easy parameter configuration using the `FlexibleConfig` class
- **Performance Profiling**: Built-in profiling tools to analyze computation bottlenecks
- **Monte Carlo Simulations**: Support for running multiple simulations simultaneously
- **Rich Visualization**: Plotting utilities for analyzing results

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sde_solver.git

# Navigate to the directory
cd sde_solver

# Install required dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start Guide

```python
import numpy as np
from sde_solver.sde_simulator import VectorizedSDESimulator

# Define drift and diffusion functions for a simple Geometric Brownian Motion
def drift(state, t):
    return 0.1 * state  # mu*X

def diffusion(state, t):
    return 0.2 * state  # sigma*X

# Create simulator
simulator = VectorizedSDESimulator(
    drift_functions=[drift],
    diffusion_functions=[diffusion]
)

# Run simulation
initial_state = np.array([1.0])
t_span = (0.0, 1.0)
t, y = simulator.run_simulation(
    initial_state=initial_state,
    t_span=t_span,
    n_steps=1000
)

# Plot results
simulator.plot_results(variable_names=["Asset Price"])
```

## Usage Examples

### Basic Example

```python
import numpy as np
from sde_solver.flexible_config import FlexibleConfig
from sde_solver.sde_system import VectorizedSDESystem
from sde_solver.sde_simulator import VectorizedSDESimulator

# Define drift and diffusion functions
def drift(state, t):
    return 1.5 * state  # Example: geometric Brownian motion drift

def diffusion(state, t):
    return 0.2 * state  # Example: geometric Brownian motion diffusion

# Create simulator
simulator = VectorizedSDESimulator(
    drift_functions=[drift],
    diffusion_functions=[diffusion],
    scheme_order=(3, 2),
    scheme_stages=3
)

# Run simulation
initial_state = np.array([100.0])  # Initial stock price
t_span = (0.0, 1.0)  # Time span from 0 to 1 year
t, y = simulator.run_simulation(
    initial_state=initial_state,
    t_span=t_span,
    n_steps=1000
)

# Plot results
simulator.plot_results(variable_names=["Stock Price"])
```

### Monte Carlo Simulation

```python
import numpy as np
from sde_solver.sde_simulator import VectorizedSDESimulator

# Define drift and diffusion functions (as in previous example)
# ...

# Create simulator
simulator = VectorizedSDESimulator(
    drift_functions=[drift],
    diffusion_functions=[diffusion],
    scheme_order=(3, 2),
    scheme_stages=3
)

# Run Monte Carlo simulation
n_simulations = 1000
initial_state = np.array([100.0])
t_span = (0.0, 1.0)
t, y_batch = simulator.run_monte_carlo(
    n_simulations=n_simulations,
    initial_state=initial_state,
    t_span=t_span,
    n_steps=1000
)

# Plot results with confidence bands
simulator.plot_results(
    variable_names=["Stock Price"],
    monte_carlo=True,
    confidence_level=0.95
)
```

## Example Implementations

The package includes several example implementations demonstrating its capabilities:

### 1. Grid Simulation

A simulation of particle movement on a 2D grid with a periodic potential. The model includes:
- Deterministic forces from a periodic potential landscape
- Spring forces from a dragging point
- Thermal noise (stochastic component)

This physically-inspired model demonstrates how to set up a 4-dimensional SDE system with position and velocity components.

```python
# Example code for grid simulation
from sde_solver.flexible_config import FlexibleConfig
from grid_simulation_script import multi_simulation_grid

# Create configuration
config = FlexibleConfig(
    k=10.0, m=1.0e-12, a=0.564, v_x=1000.0, v_y=0.0,
    T=300.0, kb=1.3807e-5, eta=20.0
)

# Run simulation
results = multi_simulation_grid(
    num_sim=16,
    pic_size=6.0,
    v_amp=1000.0,
    phi_degrees=30.0,
    dt=2.0e-8,
    Nsaved=1000
)
```

### 2. Black-Scholes Model

Implementation of the classic Black-Scholes model for option pricing. This economic model simulates stock price evolution under geometric Brownian motion assumptions:

$$dS_t = rS_t dt + \sigma S_t dW_t$$

where:
- $S_t$ is the stock price at time $t$
- $r$ is the risk-free interest rate
- $\sigma$ is the volatility
- $W_t$ is a Wiener process

```python
# Example Black-Scholes implementation
def bs_drift(state, t):
    r = 0.05  # Risk-free rate
    return r * state

def bs_diffusion(state, t):
    sigma = 0.2  # Volatility
    return sigma * state

simulator = VectorizedSDESimulator(
    drift_functions=[bs_drift],
    diffusion_functions=[bs_diffusion]
)
```

### 3. Heston Model

An advanced stochastic volatility model used in finance, extending the Black-Scholes model by making volatility itself a stochastic process:

$$dS_t = rS_t dt + \sqrt{v_t}S_t dW_t^{(1)}$$
$$dv_t = \kappa(\theta - v_t)dt + \xi\sqrt{v_t}dW_t^{(2)}$$

where:
- $S_t$ is the stock price
- $v_t$ is the variance process
- $r$ is the risk-free rate
- $\kappa$ is the rate of mean reversion
- $\theta$ is the long-term variance
- $\xi$ is the volatility of volatility
- $W_t^{(1)}$ and $W_t^{(2)}$ are correlated Wiener processes

## Project Structure

```
sde_solver/
├── __init__.py
├── grid_simulation_script.py      # Example implementation of physics grid simulation
├── black_scholes_grid_simulation.py # Example implementation of Black-Scholes model
├── heston_model_grid_simulation.py  # Example implementation of Heston model
├── final_viso.py                  # Visualization for the grid simulation 
├── benchmark_script.py            # Benchmarking utilities
├── angle_sweep_script.py          # Angular parameter sweep utility
├── setup.py                       # Package setup file
├── requirements.txt               # Package dependencies
├── .gitignore                     # Git ignore file
├── sde_solver/                    # Core package directory
│   ├── __init__.py
│   ├── config.py                  # Standard configuration class
│   ├── flexible_config.py         # Flexible parameter configuration
│   ├── noise_generation.py        # Random number generation for SDEs
│   ├── profiling.py               # Performance profiling utilities
│   ├── revised_point_generation.py # Generation of starting points
│   ├── scheme_coefficients.py     # SRK coefficient calculation
│   ├── sde_simulator.py           # High-level simulation interface
│   ├── sde_system.py              # SDE system representation
│   └── solver.py                  # SDE solver implementation
└── vectorized_imp/                # Optimized vectorized implementations
    ├── __init__.py
    ├── vectorized_solver_optimized.py
    ├── vectorized_sde_system_optimized.py
    ├── grid_simulation_optimized.py
    ├── benchmark_script.py
    └── README_vect.md
```

## Dependencies

- **NumPy** (>=1.20.0): For efficient array operations
- **SciPy** (>=1.7.0): For optimization and special functions
- **Numba** (>=0.54.0): For JIT compilation and performance acceleration
- **Matplotlib** (>=3.4.0): For visualization of results
- **Pandas** (>=1.3.0): For data manipulation and CSV export

## Performance Considerations

The library is designed with performance in mind:

- **Vectorized Operations**: Uses NumPy's efficient array operations
- **JIT Compilation**: Employs Numba to accelerate critical sections
- **Memory Efficiency**: Pre-allocates arrays to minimize allocations during simulation
- **Profiling Tools**: Includes utilities to identify and address performance bottlenecks

For optimal performance:
- Use the provided profiling tools to analyze your specific use case
- Consider batch processing for Monte Carlo simulations
- For extremely large simulations, monitor memory usage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## References

[1] Rössler, A. (2006). Runge-Kutta Methods for Itô Stochastic Differential Equations with Scalar Noise. BIT Numerical Mathematics, 46(1), 97-110. DOI: 10.1007/s10543-005-0039-7

This library implements the stochastic Runge-Kutta methods described in Rössler's paper, which provides a general framework for numerical solution of stochastic differential equations with scalar noise. The SRK coefficient schemes used in this implementation are derived from the order conditions established in this foundational work.

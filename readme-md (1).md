# SDE Solver

A high-performance library for solving Stochastic Differential Equations (SDEs) using vectorized operations and numerical methods based on stochastic Runge-Kutta schemes.

## Table of Contents

- [Introduction](#introduction)
- [Mathematical Background](#mathematical-background)
- [Features](#features)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Example Implementations](#example-implementations)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Performance Considerations](#performance-considerations)
- [Contributing](#contributing)
- [License](#license)

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

The core of this library implements Stochastic Runge-Kutta (SRK) methods for the weak approximation of SDEs. The general form of the SRK scheme for a step from $t_n$ to $t_{n+1}$ is:

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

## Usage Examples

### Basic Example

```python
import numpy as np
from sde_solver.common.flexible_config import FlexibleConfig
from sde_solver.vectorized.sde_system import VectorizedSDESystem
from sde_solver.vectorized.simulator import VectorizedSDESimulator

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
from sde_solver.vectorized.simulator import VectorizedSDESimulator

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

### 2. Black-Scholes Model

Implementation of the classic Black-Scholes model for option pricing. This economic model simulates stock price evolution under geometric Brownian motion assumptions:

$$dS_t = rS_t dt + \sigma S_t dW_t$$

where:
- $S_t$ is the stock price at time $t$
- $r$ is the risk-free interest rate
- $\sigma$ is the volatility
- $W_t$ is a Wiener process

The implementation allows for simulating price paths across a range of initial values.

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

This model demonstrates how to handle multiple stochastic components and correlation.

## Project Structure

```
sde_solver/
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── scheme_coefficients.py   # Implements SRK coefficient calculation
│   ├── flexible_config.py       # Parameter configuration management
│   └── profiling.py             # Performance profiling utilities
├── vectorized/
│   ├── __init__.py
│   ├── sde_system.py            # Vectorized SDE system representation
│   ├── solver.py                # Vectorized SDE solver implementation
│   ├── simulator.py             # High-level simulation interface
│   └── noise_generation.py      # Random number generation for SDEs
└── examples/
    └── grid_simulation.py       # Example implementation of grid simulation
```

## Dependencies

- NumPy: For efficient array operations
- SciPy: For optimization and special functions
- Numba: For JIT compilation and performance acceleration
- Matplotlib: For visualization of results
- Pandas: For data manipulation and CSV export

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

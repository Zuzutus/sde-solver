"""
SDE Simulator Package
=====================

A high-performance package for simulating stochastic differential equations (SDEs)
with vectorized operations and optimized numerical methods.

This package provides tools for:
- Defining and solving systems of SDEs
- Monte Carlo simulations
- Performance profiling
- Visualizing results
"""

__version__ = "0.1.0"

# Import core components to make them available at package level
from .config import SimulationConfig
from .flexible_config import FlexibleConfig
from .profiling import Profiler, profile_function

# Import SDE solver components
from .scheme_coefficients import SRKCoefficients
from .noise_generation import generate_noise
from .revised_point_generation import generate_starting_points

# Import vectorized implementations
from .vectorized_sde_system import VectorizedSDESystem
from .vectorized_solver import VectorizedAdaptiveSDESolver
from .vectorized_sde_simulator import VectorizedSDESimulator

# Optimized versions
from .optimized import (
    VectorizedSDESystemOptimized,
    VectorizedAdaptiveSDESolverOptimized,
    multi_simulation_grid_optimized
)

# Visualization utilities
from .visualization import (
    advanced_particle_animation,
    calculate_potential_energy,
    plot_potential_with_trajectory,
    plot_force_surface_3d,
    plot_force_2d
)

# Define what's available via import *
__all__ = [
    # Core components
    'SimulationConfig',
    'FlexibleConfig',
    'Profiler',
    'profile_function',
    
    # SDE solver components
    'SRKCoefficients',
    'generate_noise',
    'generate_starting_points',
    
    # Vectorized implementations
    'VectorizedSDESystem',
    'VectorizedAdaptiveSDESolver',
    'VectorizedSDESimulator',
    
    # Optimized versions
    'VectorizedSDESystemOptimized',
    'VectorizedAdaptiveSDESolverOptimized',
    'multi_simulation_grid_optimized',
    
    # Visualization utilities
    'advanced_particle_animation',
    'calculate_potential_energy',
    'plot_potential_with_trajectory',
    'plot_force_surface_3d',
    'plot_force_2d'
]

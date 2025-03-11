import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Optional, Dict, Tuple, Union
import time

# Import our vectorized components
from sde_solver.sde_system import VectorizedSDESystem
from sde_solver.solver import VectorizedAdaptiveSDESolver
from sde_solver.scheme_coefficients import SRKCoefficients  # Reuse existing coefficient calculator
from sde_solver.profiling import Profiler


class VectorizedSDESimulator:
    """User-friendly interface for defining and simulating systems of SDEs with vectorized operations."""

    def __init__(self,
                 drift_functions: List[Callable],
                 diffusion_functions: Optional[List[Optional[Callable]]] = None,
                 scheme_order: Tuple[int, int] = (3, 2),
                 scheme_stages: int = 3,
                 enable_profiling: bool = False):
        """
        Initialize the simulator with drift and diffusion functions.

        Parameters:
        -----------
        drift_functions : list of callables
            Each function takes (state, time) and returns a scalar for one equation
        diffusion_functions : list of callables/None, optional
            Each function takes (state, time) and returns a scalar for one equation
        scheme_order : tuple (det_order, stoch_order), default (3, 2)
            Orders of the numerical scheme
        scheme_stages : int, default 3
            Number of stages in the numerical scheme
        enable_profiling : bool, default False
            Whether to enable detailed performance profiling
        """
        # Store profiling flag
        self.enable_profiling = enable_profiling
        self.profiler = Profiler(enabled=enable_profiling)

        # Determine system dimension
        self.dim = len(drift_functions)

        # If diffusion not provided, create list of None values
        if diffusion_functions is None:
            diffusion_functions = [None] * self.dim

        # Ensure diffusion list has same length as drift list
        if len(diffusion_functions) != self.dim:
            raise ValueError(f"Number of diffusion functions ({len(diffusion_functions)}) "
                             f"must match number of drift functions ({self.dim})")

        # Create vectorized SDE system from individual functions
        with self.profiler.profile_section("system_creation"):
            self.sde_system = VectorizedSDESystem.from_functions(drift_functions, diffusion_functions)

        # Calculate scheme coefficients
        with self.profiler.profile_section("coefficients_calculation"):
            det_order, stoch_order = scheme_order
            srk = SRKCoefficients(det_order=det_order, stoch_order=stoch_order, stages=scheme_stages)
            self.coefficients = srk.calculate_coefficients()

        # Create vectorized solver
        self.solver = VectorizedAdaptiveSDESolver(self.sde_system, self.coefficients)

        # Initialize results storage
        self.results = None
        self.metadata = {}

        # Track detailed performance metrics
        self.performance_metrics = {
            'initialization_time': 0,
            'calculation_time': 0,
            'steps_per_second': 0,
            'function_calls': {
                'drift': 0,
                'diffusion': 0
            }
        }

    def run_simulation(self,
                       initial_state: np.ndarray,
                       t_span: Tuple[float, float],
                       n_steps: Optional[int] = None,
                       dt: Optional[float] = None,
                       record_every: int = 1,
                       seed: Optional[int] = None,
                       metadata: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the SDE simulation with profiling and vectorized operations.
        
        Parameters:
        -----------
        initial_state : array
            Initial state vector
        t_span : tuple (t_start, t_end)
            Time interval to solve over
        n_steps : int, optional
            Number of steps
        dt : float, optional
            Time step size
        record_every : int, default 1
            Record solution every n steps
        seed : int, optional
            Random seed for reproducibility
        metadata : dict, optional
            Additional metadata to store
            
        Returns:
        --------
        t : array
            Time points
        y : array
            Solution array, shape (n_recorded_points, system_dimension)
        """
        # Start overall profiling
        self.profiler.start()

        # Validate inputs
        if len(initial_state) != self.dim:
            raise ValueError(f"Initial state dimension ({len(initial_state)}) "
                             f"must match system dimension ({self.dim})")

        if n_steps is None and dt is None:
            # Default to 1000 steps if neither is provided
            n_steps = 1000

        if dt is not None and n_steps is None:
            # Calculate n_steps from dt
            t_start, t_end = t_span
            n_steps = int(np.ceil((t_end - t_start) / dt))

        # Record start time
        start_time = time.time()
        init_done_time = time.time()
        self.performance_metrics['initialization_time'] = init_done_time - start_time

        # Run the simulation with profiling
        with self.profiler.profile_section("solver_execution"):
            # Create an instrumented version of the SDE system's drift and diffusion functions
            original_drift_func = self.sde_system.evaluate_drift
            original_diffusion_func = self.sde_system.evaluate_diffusion

            # If profiling is enabled, wrap the functions to count calls
            if self.enable_profiling:
                def profiled_drift(state, t):
                    self.performance_metrics['function_calls']['drift'] += 1
                    return original_drift_func(state, t)

                def profiled_diffusion(state, t):
                    self.performance_metrics['function_calls']['diffusion'] += 1
                    return original_diffusion_func(state, t)

                self.sde_system.evaluate_drift = profiled_drift
                self.sde_system.evaluate_diffusion = profiled_diffusion

            # Run the simulation with vectorized solver
            t, y = self.solver.solve(
                t_span=t_span,
                y0=initial_state,
                n_steps=n_steps,
                dt=dt,
                record_every=record_every,
                seed=seed
            )

            # Restore original functions
            if self.enable_profiling:
                self.sde_system.evaluate_drift = original_drift_func
                self.sde_system.evaluate_diffusion = original_diffusion_func

        # Record end time and calculate performance metrics
        end_time = time.time()
        calculation_time = end_time - init_done_time
        execution_time = end_time - start_time

        self.performance_metrics['calculation_time'] = calculation_time
        self.performance_metrics['steps_per_second'] = n_steps / calculation_time

        # Store results
        self.results = (t, y)

        # Store metadata
        self.metadata = {
            'dimension': self.dim,
            'has_diffusion': self.sde_system.has_diffusion.tolist(),
            't_span': t_span,
            'n_steps': n_steps,
            'dt': dt if dt is not None else (t_span[1] - t_span[0]) / n_steps,
            'record_every': record_every,
            'execution_time': execution_time,
            'performance_metrics': self.performance_metrics
        }

        if metadata is not None:
            self.metadata.update(metadata)

        # Stop profiling and print summary
        self.profiler.stop(name="full_simulation")

        print(f"Simulation complete in {execution_time:.4f} seconds")
        if self.enable_profiling:
            print(f"  - Initialization: {self.performance_metrics['initialization_time']:.4f} seconds")
            print(f"  - Calculation: {self.performance_metrics['calculation_time']:.4f} seconds")
            print(f"  - Performance: {self.performance_metrics['steps_per_second']:.2f} steps/second")
            print(f"  - Drift function calls: {self.performance_metrics['function_calls']['drift']}")
            print(f"  - Diffusion function calls: {self.performance_metrics['function_calls']['diffusion']}")

        return t, y
    
    def run_monte_carlo(self,
                        n_simulations: int,
                        initial_state: Union[np.ndarray, List[np.ndarray]],
                        t_span: Tuple[float, float],
                        n_steps: Optional[int] = None,
                        dt: Optional[float] = None,
                        record_every: int = 1,
                        seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run multiple SDE simulations with the same or different initial conditions.
        Uses vectorized operations for better performance when possible.
        
        Parameters:
        -----------
        n_simulations : int
            Number of simulations to run
        initial_state : array or list of arrays
            Initial state(s) - if a single array is provided, it's used for all simulations
        t_span : tuple (t_start, t_end)
            Time interval to solve over
        n_steps : int, optional
            Number of steps
        dt : float, optional
            Time step size
        record_every : int, default 1
            Record solution every n steps
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        t : array
            Time points
        y : array, shape (n_simulations, n_recorded_points, system_dimension)
            Solution array for each simulation
        """
        # Start profiling
        self.profiler.start()
        
        # Prepare initial states
        if isinstance(initial_state, list):
            # Convert list of initial states to array
            if len(initial_state) != n_simulations:
                raise ValueError(f"Number of initial states ({len(initial_state)}) "
                                f"must match number of simulations ({n_simulations})")
            y0_batch = np.array(initial_state, dtype=np.float64)
        else:
            # Replicate the single initial state
            y0_batch = np.tile(initial_state, (n_simulations, 1))
        
        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            
        # Use batched solver if available
        try:
            with self.profiler.profile_section("batched_execution"):
                t, y_batch = self.solver.solve_batched(
                    t_span=t_span,
                    y0_batch=y0_batch,
                    n_steps=n_steps,
                    dt=dt,
                    record_every=record_every,
                    seed=seed
                )
        except (AttributeError, NotImplementedError):
            # Fall back to sequential execution if batched solving isn't available
            with self.profiler.profile_section("sequential_execution"):
                t_arr = None
                y_batch = []
                
                for i in range(n_simulations):
                    if seed is not None:
                        # Use different seeds for each simulation
                        sim_seed = seed + i
                    else:
                        sim_seed = None
                        
                    t_i, y_i = self.run_simulation(
                        initial_state=y0_batch[i],
                        t_span=t_span,
                        n_steps=n_steps,
                        dt=dt,
                        record_every=record_every,
                        seed=sim_seed
                    )
                    
                    if t_arr is None:
                        t_arr = t_i
                    y_batch.append(y_i)
                
                # Convert list to array
                t = t_arr
                y_batch = np.array(y_batch)
        
        # Stop profiling
        self.profiler.stop(name="monte_carlo_simulation")
        
        return t, y_batch
        
    def print_profiling_results(self, n=20):
        """Print the top N time-consuming functions."""
        if not self.enable_profiling:
            print("Profiling is not enabled. Initialize the simulator with enable_profiling=True.")
            return

        self.profiler.print_top_functions(n)

    def analyze_performance_bottlenecks(self):
        """Analyze and report on performance bottlenecks."""
        if not self.enable_profiling or self.results is None:
            print("Either profiling is not enabled or no simulation has been run yet.")
            return

        # Calculate function call rates
        steps = self.metadata['n_steps']
        drift_calls = self.performance_metrics['function_calls']['drift']
        diffusion_calls = self.performance_metrics['function_calls']['diffusion']

        print("\nPerformance Analysis:")
        print("---------------------")
        print(f"System dimension: {self.dim}")
        print(f"Time steps: {steps}")
        print(f"Drift function calls per step: {drift_calls / steps:.2f}")
        print(f"Diffusion function calls per step: {diffusion_calls / steps:.2f}")

        # Calculate time spent in different phases
        calc_time = self.performance_metrics['calculation_time']
        total_time = self.metadata['execution_time']

        print(f"\nTime distribution:")
        print(f"  - Setup: {100 * (total_time - calc_time) / total_time:.1f}%")
        print(f"  - Calculation: {100 * calc_time / total_time:.1f}%")

        # Suggest optimizations based on profiling data
        print("\nSuggested optimizations:")
        if drift_calls > diffusion_calls * 2:
            print("  - Optimize drift functions (called most frequently)")
        elif self.dim > 10:
            print("  - Consider more aggressive vectorization for high-dimensional system")

        # Check if there's significant noise
        has_diffusion_count = sum(self.sde_system.has_diffusion)
        if has_diffusion_count == 0:
            print("  - No diffusion terms - consider using a deterministic solver instead")
        elif has_diffusion_count < self.dim / 2:
            print(f"  - Only {has_diffusion_count}/{self.dim} equations have diffusion - consider specialized handling")

    def plot_results(self,
                     variable_names: Optional[List[str]] = None,
                     figsize: Tuple[int, int] = (12, 8),
                     plot_trajectory: bool = True,
                     monte_carlo: bool = False,
                     confidence_level: float = 0.95):
        """
        Plot the simulation results with enhanced visualization capabilities.

        Parameters:
        -----------
        variable_names : list of str, optional
            Names for each variable (for plot labels)
        figsize : tuple, default (12, 8)
            Figure size for the plots
        plot_trajectory : bool, default True
            Whether to create trajectory plots for pairs of variables
        monte_carlo : bool, default False
            Whether to plot Monte Carlo simulation results with confidence bands
        confidence_level : float, default 0.95
            Confidence level for Monte Carlo plots (between 0 and 1)
        """
        if self.results is None:
            raise ValueError("No simulation results available. Run a simulation first.")

        t, y = self.results

        # Create default variable names if not provided
        if variable_names is None:
            variable_names = [f"Variable {i + 1}" for i in range(self.dim)]

        if not monte_carlo or y.ndim == 2:
            # Regular simulation plot
            self._plot_single_simulation(t, y, variable_names, figsize, plot_trajectory)
        else:
            # Monte Carlo simulation plot
            self._plot_monte_carlo_simulation(t, y, variable_names, figsize, confidence_level)

    def _plot_single_simulation(self, t, y, variable_names, figsize, plot_trajectory):
        """Plot results from a single simulation."""
        # Create figure for time series plots
        plt.figure(figsize=figsize)

        # Calculate number of subplots needed
        n_rows = int(np.ceil(self.dim / 2))

        # Plot each variable
        for i in range(self.dim):
            plt.subplot(n_rows, 2, i + 1)
            plt.plot(t, y[:, i], label=variable_names[i])
            plt.xlabel('Time')
            plt.ylabel(variable_names[i])
            plt.legend()
            plt.grid(True)

        plt.tight_layout()

        # Create trajectory plots if requested and if we have at least 2 variables
        if plot_trajectory and self.dim >= 2:
            # Plot trajectories for selected pairs of variables
            # Only plot up to 3 trajectory plots to avoid cluttering
            pairs = [(0, 1)]  # Always plot first two variables

            if self.dim >= 4:
                pairs.append((2, 3))  # Add variables 3 & 4 if available

            if self.dim >= 6:
                pairs.append((4, 5))  # Add variables 5 & 6 if available

            plt.figure(figsize=(10, 10 if len(pairs) > 1 else 6))

            for idx, (i, j) in enumerate(pairs):
                plt.subplot(len(pairs), 1, idx + 1)
                plt.plot(y[:, i], y[:, j], '-')
                plt.xlabel(variable_names[i])
                plt.ylabel(variable_names[j])
                plt.title(f'Trajectory: {variable_names[i]} vs {variable_names[j]}')
                plt.grid(True)

            plt.tight_layout()

        plt.show()

    def _plot_monte_carlo_simulation(self, t, y_batch, variable_names, figsize, confidence_level):
        """Plot results from a Monte Carlo simulation with confidence bands."""
        # Create figure for time series plots with confidence bands
        plt.figure(figsize=figsize)

        # Calculate number of subplots needed
        n_rows = int(np.ceil(self.dim / 2))
        
        # Calculate confidence interval bounds
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        # Plot each variable
        for i in range(self.dim):
            plt.subplot(n_rows, 2, i + 1)
            
            # Calculate mean and confidence bands
            mean_trajectory = np.mean(y_batch[:, :, i], axis=0)
            lower_bound = np.percentile(y_batch[:, :, i], lower_percentile, axis=0)
            upper_bound = np.percentile(y_batch[:, :, i], upper_percentile, axis=0)
            
            # Plot mean and confidence bands
            plt.plot(t, mean_trajectory, 'b-', label=f'Mean {variable_names[i]}')
            plt.fill_between(t, lower_bound, upper_bound, color='b', alpha=0.2, 
                             label=f'{confidence_level:.0%} Confidence')
            
            # Plot a few sample trajectories for context (up to 10)
            num_samples = min(10, y_batch.shape[0])
            for j in range(num_samples):
                plt.plot(t, y_batch[j, :, i], 'r-', alpha=0.1)
            
            plt.xlabel('Time')
            plt.ylabel(variable_names[i])
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    def save_results(self,
                     basename: str = "sde_simulation",
                     formats: List[str] = ['numpy'],
                     output_dir: str = 'results'):
        """
        Save simulation results to disk.

        Parameters:
        -----------
        basename : str, default "sde_simulation"
            Base name for saved files
        formats : list of str, default ['numpy']
            File formats to save ('numpy', 'csv', 'npz')
        output_dir : str, default 'results'
            Directory to save files in
        """
        import os
        from pathlib import Path

        if self.results is None:
            raise ValueError("No simulation results available. Run a simulation first.")

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        t, y = self.results

        # Save in requested formats
        for fmt in formats:
            if fmt.lower() == 'numpy':
                # Save time and state arrays separately
                t_file = os.path.join(output_dir, f"{basename}_t.npy")
                y_file = os.path.join(output_dir, f"{basename}_y.npy")
                np.save(t_file, t)
                np.save(y_file, y)
                print(f"Saved time data to {t_file}")
                print(f"Saved state data to {y_file}")

                # Save metadata
                meta_file = os.path.join(output_dir, f"{basename}_metadata.npy")
                np.save(meta_file, self.metadata)
                print(f"Saved metadata to {meta_file}")

            elif fmt.lower() == 'npz':
                # Save everything in a single npz file
                file_path = os.path.join(output_dir, f"{basename}.npz")
                np.savez(file_path, t=t, y=y, metadata=self.metadata)
                print(f"Saved data to {file_path}")

            elif fmt.lower() == 'csv':
                # Save as CSV
                import pandas as pd

                # Create column names
                cols = ['time']
                for i in range(self.dim):
                    cols.append(f'state_{i}')

                # Create DataFrame
                data = np.column_stack([t, y])
                df = pd.DataFrame(data, columns=cols)

                # Save to CSV
                file_path = os.path.join(output_dir, f"{basename}.csv")
                df.to_csv(file_path, index=False)
                print(f"Saved CSV data to {file_path}")

                # Save metadata as JSON
                import json
                meta_file = os.path.join(output_dir, f"{basename}_metadata.json")
                with open(meta_file, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
                print(f"Saved metadata to {meta_file}")

            else:
                print(f"Unsupported format: {fmt}")

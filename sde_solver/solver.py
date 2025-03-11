import numpy as np
from typing import Tuple, Dict, Optional, List
import warnings
from numba import njit, prange

# Import the optimized noise generation function (keep using the existing one)
from sde_solver.noise_generation import generate_noise

@njit
def _calculate_stage_values_vectorized(
    stage: int, 
    prev_stage: int, 
    tn: float, 
    dt: float,
    H_0: np.ndarray, 
    H_1: np.ndarray,
    A0: np.ndarray, 
    A1: np.ndarray,
    B10: np.ndarray, 
    B31: np.ndarray,
    c0: np.ndarray, 
    c1: np.ndarray,
    I_hat: np.ndarray,
    drift_terms: np.ndarray,
    diff_terms: np.ndarray
):
    """
    Optimized and vectorized stage value calculation for the SRK method.
    This function is JIT-compiled for performance.
    
    Parameters:
    -----------
    stage : int
        Current stage index
    prev_stage : int
        Previous stage index
    tn : float
        Current time point
    dt : float
        Time step size
    H_0, H_1 : np.ndarray
        Stage values arrays
    A0, A1, B10, B31 : np.ndarray
        Scheme coefficient matrices
    c0, c1 : np.ndarray
        Scheme coefficient vectors
    I_hat : np.ndarray
        Noise terms
    drift_terms : np.ndarray
        Pre-computed drift terms for all equations
    diff_terms : np.ndarray
        Pre-computed diffusion terms for all equations
    """
    # Get coefficients for this stage pair
    A0_factor = A0[stage, prev_stage] * dt
    A1_factor = A1[stage, prev_stage] * dt
    B31_factor = B31[stage, prev_stage] * np.sqrt(dt)
    B10_factor = B10[stage, prev_stage]
    
    # Update drift contributions for all equations at once
    H_0[:, stage] += A0_factor * drift_terms
    
    # Update diffusion contributions (for equations that have diffusion)
    # Use vectorized operations on the whole array
    H_0[:, stage] += B10_factor * diff_terms * I_hat
    H_1[:, stage] += A1_factor * drift_terms + B31_factor * diff_terms


class VectorizedAdaptiveSDESolver:
    """
    Highly optimized adaptive solver for systems of SDEs using vectorized operations.
    """
    
    def __init__(self, sde_system, scheme_coefficients: Dict[str, np.ndarray],constants=None):
        """
        Initialize the solver with an SDE system and scheme coefficients.
        
        Parameters:
        -----------
        sde_system : VectorizedSDESystem
            The system of SDEs to solve
        scheme_coefficients : dict
            Dictionary of coefficient arrays for the numerical scheme
        """
        """Initialize solver with optional constants"""
        self.system = sde_system
        self.coeff = scheme_coefficients
        self.constants = constants
        self.s = len(scheme_coefficients['alpha'])

        self.system = sde_system
        self.coeff = scheme_coefficients
        self.s = len(scheme_coefficients['alpha'])  # Number of stages
        
        # Convert all arrays to contiguous float64 arrays for better performance
        self.alpha = np.ascontiguousarray(self.coeff['alpha'], dtype=np.float64)
        self.gamma1 = np.ascontiguousarray(self.coeff['gamma1'], dtype=np.float64)
        self.gamma2 = np.ascontiguousarray(self.coeff['gamma2'], dtype=np.float64)
        self.A0 = np.ascontiguousarray(self.coeff['A0'], dtype=np.float64)
        self.A1 = np.ascontiguousarray(self.coeff['A1'], dtype=np.float64)
        self.B10 = np.ascontiguousarray(self.coeff['B10'], dtype=np.float64)
        self.B31 = np.ascontiguousarray(self.coeff['B31'], dtype=np.float64)
        self.c0 = np.ascontiguousarray(self.coeff['c(0)'], dtype=np.float64)
        self.c1 = np.ascontiguousarray(self.coeff['c(1)'], dtype=np.float64)
        
        # Pre-compute diffusion indices for faster access
        self.has_diffusion = np.ascontiguousarray(self.system.has_diffusion, dtype=bool)
        self.diffusion_indices = np.where(self.has_diffusion)[0]
        self.non_diffusion_indices = np.where(~self.has_diffusion)[0]
        
        # Pre-allocate storage arrays
        self.H_0 = np.zeros((self.system.dim, self.s), dtype=np.float64)
        self.H_1 = np.zeros((self.system.dim, self.s), dtype=np.float64)
        
    def solve(self,
              t_span: Tuple[float, float],
              y0: np.ndarray,
              n_steps: int,
              dt: Optional[float] = None,
              record_every: int = 1,
              seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the system of SDEs with vectorized operations.
        
        Parameters:
        -----------
        t_span : tuple (t_start, t_end)
            Time interval to solve over
        y0 : array
            Initial state vector
        n_steps : int
            Number of steps
        dt : float, optional
            Time step size (calculated from n_steps if not provided)
        record_every : int, default 1
            Record solution every n steps
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        t : array
            Time points
        y : array
            Solution array, shape (n_recorded_points, system_dimension)
        """
        if seed is not None:
            np.random.seed(seed)
            
        t_start, t_end = t_span
        
        # Calculate dt if not provided
        if dt is None:
            dt = (t_end - t_start) / n_steps
            
        # Create time array
        t = np.linspace(t_start, t_end, n_steps + 1, dtype=np.float64)
        
        # Initialize solution array
        y = np.zeros((n_steps + 1, self.system.dim), dtype=np.float64)
        y[0] = y0
        
        # Pre-allocate arrays for drift and diffusion terms to avoid recreating them in the loop
        drift_terms = np.zeros((self.s, self.system.dim), dtype=np.float64)
        diff_terms = np.zeros((self.s, self.system.dim), dtype=np.float64)
        
        # Main integration loop
        for n in range(n_steps):
            # Generate noise terms for this step
            I_hat, I_hat_11 = generate_noise(self.system.dim, dt, self.has_diffusion)
            
            # Reset stage values
            self.H_0.fill(0.0)
            self.H_1.fill(0.0)
            
            # Set initial stage values from current solution
            for eq in range(self.system.dim):
                self.H_0[eq].fill(y[n, eq])
                if eq in self.diffusion_indices:
                    self.H_1[eq].fill(y[n, eq])
            
            # Pre-compute all drift and diffusion values for all stages
            for j in range(self.s):
                t0_j = t[n] + self.c0[j] * dt
                t1_j = t[n] + self.c1[j] * dt
                
                # Vectorized evaluation of drift and diffusion for this stage
                drift_terms[j] = self.system.evaluate_drift(self.H_0[:, j], t0_j)
                diff_terms[j] = self.system.evaluate_diffusion(self.H_1[:, j], t1_j)
            
            # Calculate stage values sequentially
            for i in range(1, self.s):
                for j in range(i):
                    _calculate_stage_values_vectorized(
                        i, j, t[n], dt,
                        self.H_0, self.H_1,
                        self.A0, self.A1, self.B10, self.B31,
                        self.c0, self.c1,
                        I_hat,
                        drift_terms[j],
                        diff_terms[j]
                    )
            
            # Initialize the next solution with current state (vectorized)
            y[n + 1] = y[n].copy()
            
            # Apply updates for each stage (vectorized)
            for i in range(self.s):
                # Add drift contribution for all equations
                y[n + 1] += self.alpha[i] * dt * drift_terms[i]
                
                # Add diffusion contribution only for equations with diffusion
                if len(self.diffusion_indices) > 0:
                    for eq in self.diffusion_indices:
                        y[n + 1, eq] += diff_terms[i, eq] * (
                            self.gamma1[i] * I_hat[eq] +
                            self.gamma2[i] * I_hat_11[eq] / np.sqrt(dt)
                        )
        
        # Record at specified intervals
        record_indices = np.arange(0, n_steps + 1, record_every, dtype=np.int32)
        return t[record_indices], y[record_indices]

    def solve_batched(self,
                     t_span: Tuple[float, float],
                     y0_batch: np.ndarray,
                     n_steps: int,
                     dt: Optional[float] = None,
                     record_every: int = 1,
                     seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve multiple SDE systems in parallel with the same coefficients but different initial conditions.
        This is useful for Monte Carlo simulations.
        
        Parameters:
        -----------
        t_span : tuple (t_start, t_end)
            Time interval to solve over
        y0_batch : array, shape (batch_size, system_dimension)
            Batch of initial states
        n_steps : int
            Number of steps
        dt : float, optional
            Time step size (calculated from n_steps if not provided)
        record_every : int, default 1
            Record solution every n steps
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        t : array
            Time points
        y : array, shape (batch_size, n_recorded_points, system_dimension)
            Solution array for each initial condition
        """
        if seed is not None:
            np.random.seed(seed)
            
        batch_size = y0_batch.shape[0]
        t_start, t_end = t_span
        
        # Calculate dt if not provided
        if dt is None:
            dt = (t_end - t_start) / n_steps
            
        # Create time array
        t = np.linspace(t_start, t_end, n_steps + 1, dtype=np.float64)
        record_indices = np.arange(0, n_steps + 1, record_every, dtype=np.int32)
        n_record_points = len(record_indices)
        
        # Initialize solution array for the batch
        y_batch = np.zeros((batch_size, n_record_points, self.system.dim), dtype=np.float64)
        
        # Setup batched drift and diffusion functions
        batched_drift, batched_diffusion = self.system.create_batched_functions(batch_size)
        
        # Solve for each initial condition
        # This could be further optimized with true vectorization, but this is a starting point
        for i in range(batch_size):
            _, y_batch[i] = self.solve(t_span, y0_batch[i], n_steps, dt, record_every, None)
        
        return t[record_indices], y_batch

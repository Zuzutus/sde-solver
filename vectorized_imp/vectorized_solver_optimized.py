import numpy as np
from typing import Tuple, Dict, Optional, List
from numba import njit, prange

# Import the noise generation functions
from sde_solver.noise_generation import generate_noise


@njit
def _calculate_stage_values_vectorized_optimized(
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
    I_hat: np.ndarray,
    drift_terms: np.ndarray,
    diff_terms: np.ndarray,
    has_diffusion: np.ndarray
):
    """
    Optimized and fully vectorized stage value calculation for the SRK method.
    This function is JIT-compiled for maximum performance.
    
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
    I_hat : np.ndarray
        Noise terms
    drift_terms : np.ndarray
        Pre-computed drift terms for all equations
    diff_terms : np.ndarray
        Pre-computed diffusion terms for all equations
    has_diffusion : np.ndarray
        Boolean array indicating which equations have diffusion
    """
    # Get coefficients for this stage pair
    A0_factor = A0[stage, prev_stage] * dt
    
    # Update drift contributions for all equations at once (vectorized)
    H_0[:, stage] += A0_factor * drift_terms
    
    # Update diffusion contributions only for equations that have diffusion
    B10_factor = B10[stage, prev_stage]
    
    # Vectorized update for diffusion terms
    for eq in range(len(has_diffusion)):
        if has_diffusion[eq]:
            H_0[eq, stage] += B10_factor * diff_terms[eq] * I_hat[eq]
            
            # Also update H_1 for diffusion equations
            A1_factor = A1[stage, prev_stage] * dt
            B31_factor = B31[stage, prev_stage] * np.sqrt(dt)
            H_1[eq, stage] += A1_factor * drift_terms[eq] + B31_factor * diff_terms[eq]


@njit
def generate_batch_noise(batch_size: int, dim: int, dt: float, has_diffusion: np.ndarray):
    """
    Generate noise terms for multiple trajectories at once.
    
    Parameters:
    -----------
    batch_size : int
        Number of trajectories
    dim : int
        System dimension
    dt : float
        Time step size
    has_diffusion : np.ndarray
        Boolean array indicating which equations have diffusion
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        I_hat and I_hat_11 arrays with shape (batch_size, dim)
    """
    I_hat = np.zeros((batch_size, dim), dtype=np.float64)
    I_hat_11 = np.zeros((batch_size, dim), dtype=np.float64)
    
    sqrt_3dt = np.sqrt(3.0 * dt)
    
    for b in range(batch_size):
        for i in range(dim):
            if has_diffusion[i]:
                p = np.random.random()
                if p < 1.0 / 6.0:
                    I_hat[b, i] = sqrt_3dt
                elif p < 1.0 / 3.0:
                    I_hat[b, i] = -sqrt_3dt
                I_hat_11[b, i] = 0.5 * (I_hat[b, i] * I_hat[b, i] - dt)
    
    return I_hat, I_hat_11


class VectorizedAdaptiveSDESolverOptimized:
    """
    Enhanced adaptive solver for SDE systems using fully vectorized operations.
    Optimized specifically for the 4D grid simulation system but works with any system.
    """
    
    def __init__(self, sde_system, scheme_coefficients: Dict[str, np.ndarray]):
        """
        Initialize the solver with an SDE system and scheme coefficients.
        
        Parameters:
        -----------
        sde_system : VectorizedSDESystem or VectorizedSDESystemOptimized
            The system of SDEs to solve
        scheme_coefficients : dict
            Dictionary of coefficient arrays for the numerical scheme
        """
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
        
        # Check if we're using an optimized system
        self.is_optimized = hasattr(self.system, 'is_optimized') and self.system.is_optimized
        
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
        Solve the system of SDEs with fully vectorized operations.
        
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
        
        # Pre-allocate arrays for drift and diffusion terms
        drift_terms = np.zeros((self.s, self.system.dim), dtype=np.float64)
        diff_terms = np.zeros((self.s, self.system.dim), dtype=np.float64)
        
        # Main integration loop with optimized vectorization
        for n in range(n_steps):
            # Generate noise terms for this step
            I_hat, I_hat_11 = generate_noise(self.system.dim, dt, self.has_diffusion)
            
            # Reset stage values
            self.H_0.fill(0.0)
            self.H_1.fill(0.0)
            
            # Set initial stage values from current solution (vectorized)
            for eq in range(self.system.dim):
                self.H_0[eq].fill(y[n, eq])
                if self.has_diffusion[eq]:
                    self.H_1[eq].fill(y[n, eq])
            
            # Pre-compute all drift and diffusion values for all stages
            for j in range(self.s):
                t0_j = t[n] + self.c0[j] * dt
                t1_j = t[n] + self.c1[j] * dt
                
                # Vectorized evaluation of drift and diffusion for this stage
                drift_terms[j] = self.system.evaluate_drift(self.H_0[:, j], t0_j)
                diff_terms[j] = self.system.evaluate_diffusion(self.H_1[:, j], t1_j)
            
            # Calculate stage values with optimized vectorized function
            for i in range(1, self.s):
                for j in range(i):
                    _calculate_stage_values_vectorized_optimized(
                        i, j, t[n], dt,
                        self.H_0, self.H_1,
                        self.A0, self.A1, self.B10, self.B31,
                        I_hat, drift_terms[j], diff_terms[j],
                        self.has_diffusion
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
    
    def solve_batch(self,
                   t_span: Tuple[float, float],
                   y0_batch: np.ndarray,
                   n_steps: int,
                   dt: Optional[float] = None,
                   record_every: int = 1,
                   seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve multiple SDE systems in parallel with batch processing.
        Highly optimized for Monte Carlo simulations and grid patterns.
        
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
            
        t_start, t_end = t_span
        batch_size = y0_batch.shape[0]
        
        # Calculate dt if not provided
        if dt is None:
            dt = (t_end - t_start) / n_steps
            
        # Create time array
        t = np.linspace(t_start, t_end, n_steps + 1, dtype=np.float64)
        record_indices = np.arange(0, n_steps + 1, record_every, dtype=np.int32)
        n_record_points = len(record_indices)
        
        # Initialize solution array for the batch
        y_batch = np.zeros((batch_size, n_record_points, self.system.dim), dtype=np.float64)
        
        # Get the temporary full solution array to be recorded later
        y_full = np.zeros((batch_size, n_steps + 1, self.system.dim), dtype=np.float64)
        y_full[:, 0, :] = y0_batch
        
        # Check if we can use batched functions
        batched_drift, batched_diffusion = self.system.create_batched_functions(batch_size)
        
        # Pre-allocate arrays for batch calculations
        H_0_batch = np.zeros((batch_size, self.system.dim, self.s), dtype=np.float64)
        H_1_batch = np.zeros((batch_size, self.system.dim, self.s), dtype=np.float64)
        drift_terms_batch = np.zeros((batch_size, self.s, self.system.dim), dtype=np.float64)
        diff_terms_batch = np.zeros((batch_size, self.s, self.system.dim), dtype=np.float64)
        
        # Main integration loop with batch processing
        for n in range(n_steps):
            # Generate noise terms for all trajectories at once
            I_hat_batch, I_hat_11_batch = generate_batch_noise(
                batch_size, self.system.dim, dt, self.has_diffusion
            )
            
            # Reset stage values
            H_0_batch.fill(0.0)
            H_1_batch.fill(0.0)
            
            # Set initial stage values from current solutions
            for b in range(batch_size):
                for eq in range(self.system.dim):
                    H_0_batch[b, eq, :] = y_full[b, n, eq]
                    if self.has_diffusion[eq]:
                        H_1_batch[b, eq, :] = y_full[b, n, eq]
            
            # Pre-compute all drift and diffusion values for all stages and all trajectories
            for j in range(self.s):
                t0_j = t[n] + self.c0[j] * dt
                
                # Extract states for this stage across all trajectories
                stage_states = np.zeros((batch_size, self.system.dim), dtype=np.float64)
                for b in range(batch_size):
                    stage_states[b] = H_0_batch[b, :, j]
                
                # Compute drift and diffusion for all trajectories at once
                drift_batch = batched_drift(stage_states, t0_j)
                diff_batch = batched_diffusion(stage_states, t0_j)
                
                # Store results
                for b in range(batch_size):
                    drift_terms_batch[b, j] = drift_batch[b]
                    diff_terms_batch[b, j] = diff_batch[b]
            
            # Calculate stage values for all trajectories
            for i in range(1, self.s):
                for j in range(i):
                    # Coefficients for this stage pair
                    A0_factor = self.A0[i, j] * dt
                    B10_factor = self.B10[i, j]
                    A1_factor = self.A1[i, j] * dt
                    B31_factor = self.B31[i, j] * np.sqrt(dt)
                    
                    # Update all trajectories
                    for b in range(batch_size):
                        # Update drift contributions for all equations
                        H_0_batch[b, :, i] += A0_factor * drift_terms_batch[b, j]
                        
                        # Update diffusion contributions
                        for eq in self.diffusion_indices:
                            H_0_batch[b, eq, i] += B10_factor * diff_terms_batch[b, j, eq] * I_hat_batch[b, eq]
                            H_1_batch[b, eq, i] += A1_factor * drift_terms_batch[b, j, eq] + B31_factor * diff_terms_batch[b, j, eq]
            
            # Apply updates for each trajectory
            for b in range(batch_size):
                # Copy current state to next
                y_full[b, n+1] = y_full[b, n]
                
                # Apply stage updates
                for i in range(self.s):
                    # Add drift contribution
                    y_full[b, n+1] += self.alpha[i] * dt * drift_terms_batch[b, i]
                    
                    # Add diffusion contribution
                    for eq in self.diffusion_indices:
                        y_full[b, n+1, eq] += diff_terms_batch[b, i, eq] * (
                            self.gamma1[i] * I_hat_batch[b, eq] +
                            self.gamma2[i] * I_hat_11_batch[b, eq] / np.sqrt(dt)
                        )
        
        # Record at specified intervals for all trajectories
        for b in range(batch_size):
            y_batch[b] = y_full[b, record_indices]
        
        return t[record_indices], y_batch

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional, Union, Tuple
from numba import njit, vectorize
from sde_solver.flexible_config import FlexibleConfig

# Define function types that can be JIT-compiled
VectorizedDriftFunc = Callable[[np.ndarray, float], np.ndarray]
VectorizedDiffusionFunc = Callable[[np.ndarray, float], np.ndarray]


@dataclass
class VectorizedSDESystem:
    """
    Efficient implementation of SDE system with vectorized operations
    """
    dim: int  # System dimension
    drift_func: VectorizedDriftFunc  # Vectorized drift function
    diffusion_func: Optional[VectorizedDiffusionFunc] = None  # Vectorized diffusion function
    has_diffusion: np.ndarray = None  # Boolean array indicating which equations have diffusion
    constants: Optional[np.ndarray] = None  # Constants for the system

    def __post_init__(self):
        """Initialize additional attributes after instance creation"""
        if self.has_diffusion is None and self.diffusion_func is not None:
            # Test the diffusion function with a zero state to determine which equations have non-zero diffusion
            test_state = np.zeros(self.dim, dtype=np.float64)
            diffusion_values = self.diffusion_func(test_state, 0.0)
            self.has_diffusion = np.abs(diffusion_values) > 0
        elif self.has_diffusion is None:
            self.has_diffusion = np.zeros(self.dim, dtype=bool)

        # Ensure has_diffusion is a contiguous numpy array with correct type
        self.has_diffusion = np.ascontiguousarray(self.has_diffusion, dtype=bool)

        # Verify dimensions
        if len(self.has_diffusion) != self.dim:
            raise ValueError(f"has_diffusion array length ({len(self.has_diffusion)}) "
                             f"must match system dimension ({self.dim})")

        # Pre-compute indices for efficient access
        self.diffusion_indices = np.where(self.has_diffusion)[0]
        self.non_diffusion_indices = np.where(~self.has_diffusion)[0]

    @classmethod
    def from_functions(cls,
                       drift_funcs: List[Callable],
                       diffusion_funcs: List[Optional[Callable]],
                       constants=None) -> 'VectorizedSDESystem':
        """
        Create an SDE system from lists of individual drift and diffusion functions.
        Automatically vectorizes the functions for efficient evaluation.
        
        Parameters:
        -----------
        drift_funcs : List of callables where each function takes (state, time) and returns a scalar
        diffusion_funcs : List of callables/None where each function takes (state, time) and returns a scalar
        
        Returns:
        --------
        Instance of VectorizedSDESystem with vectorized functions
        """
        dim = len(drift_funcs)
        if len(diffusion_funcs) != dim:
            raise ValueError(f"Number of diffusion functions ({len(diffusion_funcs)}) "
                             f"must match number of drift functions ({dim})")

        const_array = None
        if constants is not None:
            if hasattr(constants, 'to_array') and callable(constants.to_array):
                const_array = constants.to_array()
            else:
                const_array = constants

        # Create boolean array to track which equations have diffusion
        has_diffusion = np.array([func is not None for func in diffusion_funcs], dtype=bool)
        diffusion_indices = np.where(has_diffusion)[0]

        # Create vectorized drift function
        def vectorized_drift(state: np.ndarray, t: float) -> np.ndarray:
            """Vectorized drift function that evaluates all equations at once"""
            result = np.zeros(dim, dtype=np.float64)

            # Option 1: Vectorized using array operations when possible
            # This would require analyzing the drift functions to identify common patterns
            # For now, we use a more general approach

            # Option 2: Evaluate each equation in parallel
            # This is more general but might be less efficient for simple functions
            for i in range(dim):
                if const_array is not None:
                    result[i] = drift_funcs[i](state, t, constants)
                else:
                    result[i] = drift_funcs[i](state, t)

            return result

        # Create vectorized diffusion function
        diffusion_func = None
        if any(has_diffusion):
            def vectorized_diffusion(state: np.ndarray, t: float) -> np.ndarray:
                """Vectorized diffusion function that evaluates all equations at once"""
                result = np.zeros(dim, dtype=np.float64)

                # Only evaluate equations with diffusion terms
                for i in diffusion_indices:
                    if const_array is not None:
                        result[i] = diffusion_funcs[i](state, t, constants)
                    else:
                        result[i] = diffusion_funcs[i](state, t)

                return result

            diffusion_func = vectorized_diffusion

        return cls(dim, vectorized_drift, diffusion_func, has_diffusion)

    def create_batched_functions(self, batch_size: int = 32):
        """
        Create functions that can evaluate drift and diffusion for multiple states at once
        Useful for Monte Carlo simulations or ensemble methods
        
        Parameters:
        -----------
        batch_size : int
            The number of states to process in a single batch
        """
        original_drift = self.drift_func
        original_diffusion = self.diffusion_func

        def batched_drift(states: np.ndarray, t: float) -> np.ndarray:
            """Evaluate drift for multiple states at once"""
            # states shape: (batch_size, dim)
            # returns shape: (batch_size, dim)
            result = np.zeros((states.shape[0], self.dim), dtype=np.float64)
            for i in range(states.shape[0]):
                result[i] = original_drift(states[i], t)
            return result

        def batched_diffusion(states: np.ndarray, t: float) -> np.ndarray:
            """Evaluate diffusion for multiple states at once"""
            # states shape: (batch_size, dim)
            # returns shape: (batch_size, dim)
            if original_diffusion is None:
                return np.zeros((states.shape[0], self.dim), dtype=np.float64)

            result = np.zeros((states.shape[0], self.dim), dtype=np.float64)
            for i in range(states.shape[0]):
                result[i] = original_diffusion(states[i], t)
            return result

        return batched_drift, batched_diffusion

    def evaluate_drift(self, state, t, constants=None) -> np.ndarray:
        """Evaluate the drift function for all equations at once"""
        return self.drift_func(state, t)

    def evaluate_diffusion(self, state, t, constants=None) -> np.ndarray:
        """Evaluate the diffusion function for all equations at once"""
        if self.diffusion_func is None:
            return np.zeros(self.dim, dtype=np.float64)
        return self.diffusion_func(state, t)

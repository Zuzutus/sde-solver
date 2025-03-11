import numpy as np
from numba import njit
from typing import Callable, List, Optional, Union, Tuple

# Import base class for extension
from sde_solver.vectorized_sde_system import VectorizedSDESystem


@njit
def optimized_drift(state: np.ndarray, t: float, constants: np.ndarray) -> np.ndarray:
    """
    Optimized vectorized drift function for the 4D system in grid_simulation_script.py.
    This calculates all drift components at once using array operations.
    
    Parameters:
    -----------
    state : np.ndarray
        State vector [x_pos, x_vel, y_pos, y_vel]
    t : float
        Current time
    constants : np.ndarray
        Constants array from FlexibleConfig
        
    Returns:
    --------
    np.ndarray
        Drift values for all 4 equations
    """
    # Initialize result array
    result = np.zeros(4, dtype=np.float64)
    
    # Extract state variables
    x_pos, x_vel, y_pos, y_vel = state
    
    # First set the simple position drift terms (equals velocity)
    result[0] = x_vel  # drift1: x position drift
    result[2] = y_vel  # drift3: y position drift
    
    # Extract constants
    k = constants[0]
    m = constants[1]
    a = constants[2]
    eta = constants[5]
    v_x = constants[6]
    v_y = constants[7]
    c_x = constants[8]  # Starting position X
    c_y = constants[9]  # Starting position Y
    
    # Common calculations - precompute once for both equations
    mu = 2.0 * np.sqrt(k / m)
    u0 = eta * (a * a) * k / (4.0 * np.pi * np.pi)
    sqrt3 = np.sqrt(3.0)
    
    # Calculate drift2 (x velocity)
    term1_x = -mu * x_vel  # Damping
    term2_x = (4.0 * np.pi * u0 / (m * a)) * np.sin(2.0 * np.pi * x_pos / a) * \
            np.cos(2.0 * np.pi * y_pos / (a * sqrt3))  # Periodic potential
    term3_x = (k / m) * (t * v_x + c_x - x_pos)  # Spring force
    result[1] = term1_x + term2_x + term3_x
    
    # Calculate drift4 (y velocity)
    term1_y = -mu * y_vel  # Damping
    term2_y = (4.0 * np.pi * u0 / (m * a * sqrt3)) * (
            np.sin(4.0 * np.pi * y_pos / (a * sqrt3)) +
            np.sin(2.0 * np.pi * y_pos / (a * sqrt3)) * np.cos(2.0 * np.pi * x_pos / a)
    )  # Periodic potential
    term3_y = (k / m) * (t * v_y + c_y - y_pos)  # Spring force
    result[3] = term1_y + term2_y + term3_y
    
    return result


@njit
def optimized_diffusion(state: np.ndarray, t: float, constants: np.ndarray) -> np.ndarray:
    """
    Optimized vectorized diffusion function for the 4D system.
    Only applies thermal noise to velocity components.
    
    Parameters:
    -----------
    state : np.ndarray
        State vector [x_pos, x_vel, y_pos, y_vel]
    t : float
        Current time
    constants : np.ndarray
        Constants array from FlexibleConfig
    """
    # Initialize result - no diffusion for position components
    result = np.zeros(4, dtype=np.float64)
    
    # Extract constants
    k = constants[0]
    m = constants[1]
    kb = constants[4]
    T = constants[3]
    
    # Calculate thermal noise once (same for both velocity components)
    mu = 2.0 * np.sqrt(k / m)
    noise_amplitude = np.sqrt(2.0 * kb * T * mu / m)
    
    # Apply noise only to velocity components
    result[1] = noise_amplitude  # x velocity diffusion
    result[3] = noise_amplitude  # y velocity diffusion
    
    return result


@njit
def optimized_drift_batch(states: np.ndarray, t: float, constants: np.ndarray) -> np.ndarray:
    """
    Optimized vectorized drift function that processes multiple states at once.
    Perfect for Monte Carlo simulations with many trajectories.
    
    Parameters:
    -----------
    states : np.ndarray with shape (batch_size, 4)
        Batch of state vectors [x_pos, x_vel, y_pos, y_vel]
    t : float
        Current time
    constants : np.ndarray
        Constants array from FlexibleConfig
        
    Returns:
    --------
    np.ndarray with shape (batch_size, 4)
        Drift values for all states and equations
    """
    batch_size = states.shape[0]
    result = np.zeros((batch_size, 4), dtype=np.float64)
    
    # Extract constants
    k = constants[0]
    m = constants[1]
    a = constants[2]
    eta = constants[5]
    v_x = constants[6]
    v_y = constants[7]
    c_x = constants[8]
    c_y = constants[9]
    
    # Common calculations - precompute once
    mu = 2.0 * np.sqrt(k / m)
    u0 = eta * (a * a) * k / (4.0 * np.pi * np.pi)
    sqrt3 = np.sqrt(3.0)
    
    # Extract state variables using array indexing
    x_pos = states[:, 0]
    x_vel = states[:, 1]
    y_pos = states[:, 2]
    y_vel = states[:, 3]
    
    # Set position drift = velocity for all states at once
    result[:, 0] = x_vel  # drift1: x position drift  
    result[:, 2] = y_vel  # drift3: y position drift
    
    # Calculate x velocity drift (drift2) for all states at once
    result[:, 1] = (-mu * x_vel + 
                    (4.0 * np.pi * u0 / (m * a)) * np.sin(2.0 * np.pi * x_pos / a) * 
                    np.cos(2.0 * np.pi * y_pos / (a * sqrt3)) + 
                    (k / m) * (t * v_x + c_x - x_pos))
    
    # Calculate y velocity drift (drift4) for all states at once
    result[:, 3] = (-mu * y_vel + 
                    (4.0 * np.pi * u0 / (m * a * sqrt3)) * (
                        np.sin(4.0 * np.pi * y_pos / (a * sqrt3)) +
                        np.sin(2.0 * np.pi * y_pos / (a * sqrt3)) * np.cos(2.0 * np.pi * x_pos / a)
                    ) + 
                    (k / m) * (t * v_y + c_y - y_pos))
    
    return result


@njit
def optimized_diffusion_batch(states: np.ndarray, t: float, constants: np.ndarray) -> np.ndarray:
    """
    Optimized vectorized diffusion function for batch processing.
    
    Parameters:
    -----------
    states : np.ndarray with shape (batch_size, 4)
        Batch of state vectors
    t : float
        Current time
    constants : np.ndarray
        Constants array from FlexibleConfig
    """
    batch_size = states.shape[0]
    result = np.zeros((batch_size, 4), dtype=np.float64)
    
    # Extract constants (same for all states)
    k = constants[0]
    m = constants[1]
    kb = constants[4]
    T = constants[3]
    
    # Calculate thermal noise (same for all states)
    mu = 2.0 * np.sqrt(k / m)
    noise_amplitude = np.sqrt(2.0 * kb * T * mu / m)
    
    # Set diffusion for velocity components
    result[:, 1] = noise_amplitude  # x velocity diffusion
    result[:, 3] = noise_amplitude  # y velocity diffusion
    
    return result


class VectorizedSDESystemOptimized(VectorizedSDESystem):
    """
    Enhanced version of VectorizedSDESystem with fully vectorized operations.
    Optimized specifically for the 4D grid simulation system.
    """
    
    @classmethod
    def from_functions(cls,
                      drift_funcs: List[Callable],
                      diffusion_funcs: List[Optional[Callable]],
                      constants=None) -> 'VectorizedSDESystemOptimized':
        """
        Create an SDE system from lists of individual drift and diffusion functions.
        Automatically detects the 4D system and uses optimized vectorized operations.
        
        Parameters:
        -----------
        drift_funcs : List of callables
            Each function takes (state, time) and returns a scalar for one equation
        diffusion_funcs : List of callables/None
            Each function takes (state, time) and returns a scalar for one equation
        constants : array or FlexibleConfig, optional
            Constants for the system
            
        Returns:
        --------
        Instance of VectorizedSDESystemOptimized with vectorized functions
        """
        dim = len(drift_funcs)
        if len(diffusion_funcs) != dim:
            raise ValueError(f"Number of diffusion functions ({len(diffusion_funcs)}) "
                             f"must match number of drift functions ({dim})")

        # Convert constants to array if needed
        const_array = None
        if constants is not None:
            if hasattr(constants, 'to_array') and callable(constants.to_array):
                const_array = constants.to_array()
            else:
                const_array = constants

        # Create boolean array to track which equations have diffusion
        has_diffusion = np.array([func is not None for func in diffusion_funcs], dtype=bool)
        
        # Check if this is the specific 4D system from grid_simulation_script.py
        is_grid_simulation = False
        if dim == 4 and const_array is not None:
            # Check if the drift functions match our expected pattern
            if (hasattr(drift_funcs[0], '__code__') and hasattr(drift_funcs[2], '__code__')):
                # Check for velocity-position relationships
                fn1_str = drift_funcs[0].__code__.co_consts
                fn3_str = drift_funcs[2].__code__.co_consts
                if any("state[1]" in str(c) for c in fn1_str) and any("state[3]" in str(c) for c in fn3_str):
                    is_grid_simulation = True
        
        # Choose vectorization strategy
        if is_grid_simulation:
            # Use optimized vectorized functions for the specific 4D system
            print("Using optimized vectorized functions for 4D grid simulation system")
            
            def vectorized_drift(state: np.ndarray, t: float) -> np.ndarray:
                """Optimized vectorized drift function for the 4D grid simulation system"""
                return optimized_drift(state, t, const_array)
                
            # Optimized diffusion function if any equations have diffusion
            diffusion_func = None
            if any(has_diffusion):
                def vectorized_diffusion(state: np.ndarray, t: float) -> np.ndarray:
                    """Optimized vectorized diffusion function for the 4D grid simulation system"""
                    return optimized_diffusion(state, t, const_array)
                    
                diffusion_func = vectorized_diffusion
                
        else:
            # Use standard approach for other systems
            print("Using standard implementation for non-grid simulation system")
            
            def vectorized_drift(state: np.ndarray, t: float) -> np.ndarray:
                """General vectorized drift function for any system"""
                result = np.zeros(dim, dtype=np.float64)
                
                # Evaluate each equation in a loop
                for i in range(dim):
                    if const_array is not None:
                        result[i] = drift_funcs[i](state, t, const_array)
                    else:
                        result[i] = drift_funcs[i](state, t)
                
                return result
            
            diffusion_func = None
            if any(has_diffusion):
                def vectorized_diffusion(state: np.ndarray, t: float) -> np.ndarray:
                    """General vectorized diffusion function for any system"""
                    result = np.zeros(dim, dtype=np.float64)
                    
                    # Only evaluate equations with diffusion terms
                    for i in range(dim):
                        if has_diffusion[i]:
                            if const_array is not None:
                                result[i] = diffusion_funcs[i](state, t, const_array)
                            else:
                                result[i] = diffusion_funcs[i](state, t)
                    
                    return result
                
                diffusion_func = vectorized_diffusion
                
        # Create the system with the appropriate vectorized functions
        instance = super(VectorizedSDESystemOptimized, cls).__new__(cls)
        instance.dim = dim
        instance.drift_func = vectorized_drift
        instance.diffusion_func = diffusion_func
        instance.has_diffusion = np.ascontiguousarray(has_diffusion, dtype=bool)
        instance.constants = const_array
        
        # Pre-compute indices for efficient access
        instance.diffusion_indices = np.where(instance.has_diffusion)[0]
        instance.non_diffusion_indices = np.where(~instance.has_diffusion)[0]
        
        # Flag to indicate whether we're using optimized functions
        instance.is_optimized = is_grid_simulation
        
        return instance
    
    def create_batched_functions(self, batch_size=32):
        """
        Create functions that evaluate drift and diffusion for multiple states at once.
        This is important for Monte Carlo simulations.
        
        Parameters:
        -----------
        batch_size : int
            Number of states to process simultaneously
            
        Returns:
        --------
        tuple:
            (batched_drift_func, batched_diffusion_func)
        """
        # If using optimized functions for the 4D grid simulation
        if hasattr(self, 'is_optimized') and self.is_optimized and self.constants is not None:
            def batched_drift(states, t):
                return optimized_drift_batch(states, t, self.constants)
                
            def batched_diffusion(states, t):
                if any(self.has_diffusion):
                    return optimized_diffusion_batch(states, t, self.constants)
                return np.zeros((states.shape[0], self.dim), dtype=np.float64)
                
            return batched_drift, batched_diffusion
        else:
            # Fall back to standard implementation for non-optimized systems
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

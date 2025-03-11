# Vectorized Implementation of SDE Solver

This folder contains optimized implementations that focus on vectorizing function calculations for SDE systems. The key difference is in **how the drift and diffusion functions are written** to leverage vectorized operations.

## How Function Implementation Differs

### Standard Function Implementation (Non-Vectorized)
```python
# Standard approach - operates on a single state
def drift2(state, t, constants=None):
    """Drift function for x velocity"""
    # Unpack state variables individually
    x1, x2, y1, _ = state
    
    # Calculate terms one by one
    term1 = -mu * x2  # Damping
    term2 = (4.0 * np.pi * u0 / (m * a)) * np.sin(2.0 * np.pi * x1 / a) * \
            np.cos(2.0 * np.pi * y1 / (a * np.sqrt(3.0)))  # Periodic potential
    term3 = (k / m) * (t * v_x + c_x - x1)  # Spring force
    
    return term1 + term2 + term3
```

### Vectorized Function Implementation
```python
# Vectorized approach - operates on arrays of states
@njit
def optimized_drift_batch(states, t, constants):
    """Process multiple states at once"""
    batch_size = states.shape[0]
    result = np.zeros((batch_size, 4), dtype=np.float64)
    
    # Extract all state variables using array indexing
    x_pos = states[:, 0]  # All x positions at once
    x_vel = states[:, 1]  # All x velocities at once
    y_pos = states[:, 2]  # All y positions at once
    y_vel = states[:, 3]  # All y velocities at once
    
    # Calculate terms for all states at once using array operations
    result[:, 1] = (-mu * x_vel + 
                  (4.0 * np.pi * u0 / (m * a)) * np.sin(2.0 * np.pi * x_pos / a) * 
                  np.cos(2.0 * np.pi * y_pos / (a * sqrt3)) + 
                  (k / m) * (t * v_x + c_x - x_pos))
    
    return result
```

## Key Differences When Writing Vectorized Functions

1. **Array Operations Instead of Scalar Operations**
   - Work with entire arrays (rows, columns) instead of individual elements
   - Use NumPy's vectorized operations like `np.sin()` applied to entire arrays

2. **Shape Handling**
   - Standard: Functions operate on a single state vector (shape: `(dim,)`)
   - Vectorized: Functions operate on batches of state vectors (shape: `(batch_size, dim)`)

3. **Numba JIT Compilation**
   - Add `@njit` decorator to speed up computation-intensive functions
   - Ensure functions contain only Numba-compatible operations

4. **Pre-compute Values Once**
   ```python
   # Instead of calculating for each state in a loop:
   for i in range(batch_size):
       mu = 2.0 * np.sqrt(k / m)  # Calculated repeatedly
   
   # Calculate once for all states:
   mu = 2.0 * np.sqrt(k / m)  # Calculated only once
   ```

5. **Avoid Loops When Possible**
   ```python
   # Instead of:
   for i in range(batch_size):
       result[i, 1] = -mu * states[i, 1]
   
   # Use array operations:
   result[:, 1] = -mu * states[:, 1]
   ```

## Steps to Vectorize Your System

1. **Identify Array Operations**
   - Look for calculations you can apply to entire arrays at once
   - Use broadcasting to apply operations between arrays of different shapes

2. **Separate Position/Velocity Functions**
   - Sometimes it's cleaner to handle position and velocity components separately
   - In the examples, velocity equations have diffusion while position equations don't

3. **Create Batched Functions**
   - Write specialized functions that handle multiple states at once
   - These are crucial for Monte Carlo simulations

4. **Return Optimized for Multiple Dimensions**
   - Standard: Return a single value or 1D array
   - Vectorized: Return a 2D array with values for all states

By structuring your drift and diffusion functions in this vectorized way, you can achieve significant performance improvements, especially for large-scale simulations with many equations or trajectories.
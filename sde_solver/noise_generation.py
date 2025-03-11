import numpy as np
from numba import jit, float64, boolean, int32



@jit(nopython=True)
def generate_noise(dim: int, dt: float, has_diffusion: np.ndarray):
    """Generate noise terms for each equation that has diffusion"""
    I_hat = np.zeros(dim, dtype=np.float64)
    I_hat_11 = np.zeros(dim, dtype=np.float64)

    sqrt_3dt = np.sqrt(3.0 * dt)
    for i in range(dim):
        if has_diffusion[i]:
            p = np.random.random()
            if p < 1.0 / 6.0:
                I_hat[i] = sqrt_3dt
            elif p < 1.0 / 3.0:
                I_hat[i] = -sqrt_3dt
            I_hat_11[i] = 0.5 * (I_hat[i] * I_hat[i] - dt)

    return I_hat, I_hat_11

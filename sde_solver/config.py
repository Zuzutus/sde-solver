from dataclasses import dataclass
import numpy as np


@dataclass
class SimulationConfig:
    k: float = 10.0
    m: float = 1.0e-12
    a: float = 0.564
    v_x: float = 1000.0
    v_y: float = 0.0
    T: float = 300.0
    kb: float = 1.3807e-5
    eta: float = 20.0

    @property
    def mu(self) -> float:
        return 2.0 * np.sqrt(self.k / self.m)

    def to_array(self) -> np.ndarray:
        """Convert to array for numba functions"""
        return np.array([
            self.k, self.m, self.a, self.mu, self.eta,
            self.kb, self.T, self.v_x, self.v_y, 0.0, 0.0
        ], dtype=np.float64)
import numpy as np
from scipy.optimize import root, fsolve
import warnings
from numba import jit
from typing import Tuple, Dict


# Helper functions for Numba optimization
@jit(nopython=True)
def _compute_B31e_terms(B31: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute commonly used B31 terms with Numba optimization"""
    B31e = np.sum(B31, axis=1)
    B31e_squared = B31e * B31e  # Faster than B31e ** 2
    B31_B31e = np.dot(B31, B31e)
    return B31e, B31e_squared, B31_B31e


@jit(nopython=True)
def _compute_B10e_terms(B10: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute commonly used B10 terms with Numba optimization"""
    B10e = np.sum(B10, axis=1)
    B10e_squared = B10e * B10e
    return B10e, B10e_squared


@jit(nopython=True)
def _compute_matrix_conditions(alpha: np.ndarray, gamma1: np.ndarray, gamma2: np.ndarray,
                               A0: np.ndarray, A1: np.ndarray, B10: np.ndarray, B31: np.ndarray) -> np.ndarray:
    """Compute all matrix conditions efficiently using Numba"""
    conditions = np.zeros(28, dtype=np.float64)

    # Compute commonly used terms once
    gamma1_sum = np.sum(gamma1)
    gamma2_sum = np.sum(gamma2)
    A0e = np.sum(A0, axis=1)
    A1e = np.sum(A1, axis=1)

    # Get precomputed terms
    B31e, B31e_squared, B31_B31e = _compute_B31e_terms(B31)
    B10e, B10e_squared = _compute_B10e_terms(B10)

    # Fill conditions array
    conditions[0] = np.sum(alpha) - 1.0  # α^T e = 1
    conditions[1] = gamma2_sum  # γ^(2)T e = 0
    conditions[2] = gamma1_sum * gamma1_sum - 1.0  # (γ^(1)T e)^2 = 1
    conditions[3] = np.dot(gamma1, B31e)  # γ^(1)T B^(3)(1)e = 0
    conditions[4] = np.dot(gamma2, B31_B31e)  # γ^(2)T (B^(3)(1)(B^(3)(1)e)) = 0

    # Calculate intermediate terms for remaining conditions
    B31_B31_B31e = np.dot(B31, B31_B31e)
    A1_B10e = np.dot(A1, B10e)
    B10_B31e = np.dot(B10, B31e)

    # Remaining conditions
    conditions[5] = np.dot(gamma1, B31_B31_B31e)  # γ^(1)T (B^(3)(1)(B^(3)(1)(B^(3)(1)e)))
    conditions[6] = np.dot(alpha, A0e) - 0.5  # α^T A^(0)e = 1/2
    conditions[7] = np.dot(gamma1, B31e * A1e)  # γ^(1)T ((B^(3)(1)e)(A^(1)e))
    conditions[8] = np.dot(gamma1, np.dot(B31, B31e_squared))  # γ^(1)T (B^(3)(1)(B^(3)(1)e)^2)
    conditions[9] = np.dot(gamma1, B31_B31e)  # γ^(1)T (B^(3)(1)(B^(3)(1)e))
    conditions[10] = np.dot(gamma1, B31e * A1e)  # (γ^(1)T e)(γ^(1)T (B^(3)(1)e)^2)
    conditions[11] = np.dot(gamma1, np.dot(B31, A1_B10e))  # γ^(1)T (B^(3)(1)(A^(1)(B^(1)(0)e)))
    conditions[12] = np.dot(alpha, B10e * B10_B31e)  # α^T ((B^(1)(0)e)(B^(1)(0)(B^(3)(1)e)))
    conditions[13] = np.dot(gamma1, B31e * A1_B10e)  # γ^(1)T ((B^(3)(1)e)(A^(1)(B^(1)(0)e)))
    conditions[14] = np.dot(gamma1, np.dot(A1, B10_B31e))  # γ^(1)T (A^(1)(B^(1)(0)(B^(3)(1)e)))
    conditions[15] = np.dot(gamma1, B31e * B31_B31e)  # γ^(1)T ((B^(3)(1)e)(B^(3)(1)(B^(3)(1)e)))
    conditions[16] = np.dot(gamma1, B31e * B31e * B31e)  # γ^(1)T (B^(3)(1)e)^3
    conditions[17] = np.dot(gamma1, A1_B10e)  # γ^(1)T (A^(1)(B^(1)(0)e))
    conditions[18] = np.dot(alpha, B10_B31e)  # α^T (B^(1)(0)(B^(3)(1)e))
    conditions[19] = np.dot(gamma1, np.dot(B31, A1e))  # γ^(1)T (B^(3)(1)(A^(1)e))
    conditions[20] = gamma1_sum * np.dot(alpha, B10e) - 0.5  # (γ^(1)T e)(α^T B^(1)(0)e)
    conditions[21] = np.dot(gamma2, A1e)  # γ^(2)T A^(1)e
    conditions[22] = np.dot(alpha, B10e_squared) - 0.5  # α^T (B^(1)(0)e)^2
    conditions[23] = np.dot(gamma2, B31e_squared)  # γ^(2)T (B^(3)(1)e)^2
    conditions[24] = np.dot(gamma2, np.dot(A1, B10e_squared))  # γ^(2)T (A^(1)(B^(1)(0)e)^2)
    conditions[25] = np.dot(gamma2, B31e) - 1.0  # γ^(2)T B^(3)(1)e = 1
    conditions[26] = gamma1_sum * np.dot(gamma1, A1e) - 0.5  # (γ^(1)T e)(γ^(1)T A^(1)e)
    conditions[27] = np.dot(gamma2, A1_B10e)  # γ^(2)T (A^(1)(B^(1)(0)e))

    return conditions


@jit(nopython=True)
def _compute_deterministic_conditions(alpha: np.ndarray, A0: np.ndarray) -> np.ndarray:
    """Compute deterministic order conditions with Numba optimization"""
    A0e = np.sum(A0, axis=1)
    A0_A0e = np.dot(A0, A0e)

    conditions = np.zeros(2, dtype=np.float64)
    conditions[0] = np.dot(alpha, A0_A0e) - 1.0 / 6.0
    conditions[1] = np.dot(alpha, A0e * A0e) - 1.0 / 3.0

    return conditions


class SRKCoefficients:
    def __init__(self, det_order: int, stoch_order: int, stages: int):
        if det_order not in [1, 2, 3]:
            raise ValueError("Deterministic order must be 1, 2, or 3")
        if stoch_order not in [1, 2]:
            raise ValueError("Stochastic order must be 1 or 2")
        if stages < 3:
            raise ValueError("At least 3 stages recommended")

        self.det_order = det_order
        self.stoch_order = stoch_order
        self.stages = stages

        # Initialize coefficient arrays with explicit dtypes
        self.alpha = np.zeros(stages, dtype=np.float64)
        self.gamma1 = np.zeros(stages, dtype=np.float64)
        self.gamma2 = np.zeros(stages, dtype=np.float64)
        self.A0 = np.zeros((stages, stages), dtype=np.float64)
        self.A1 = np.zeros((stages, stages), dtype=np.float64)
        self.B10 = np.zeros((stages, stages), dtype=np.float64)
        self.B31 = np.zeros((stages, stages), dtype=np.float64)
        self.c0 = np.zeros(stages, dtype=np.float64)
        self.c1 = np.zeros(stages, dtype=np.float64)

    def _generate_initial_guess(self) -> np.ndarray:
        """Generate an optimized initial guess"""
        s = self.stages
        num_lower_triangular = (s * (s - 1)) // 2

        # Initialize with explicit dtypes
        alpha = np.ones(s, dtype=np.float64) / s
        gamma1 = np.zeros(s, dtype=np.float64)
        gamma1[0] = np.sqrt(0.5)
        gamma2 = np.zeros(s, dtype=np.float64)
        gamma2[1:] = np.array([1.0 / (s - 1) if i % 2 == 0 else -1.0 / (s - 1)
                               for i in range(s - 1)], dtype=np.float64)

        # Initialize matrix elements
        matrix_elements = np.ones(4 * num_lower_triangular, dtype=np.float64) / s

        return np.concatenate([alpha, gamma1, gamma2, matrix_elements])

    def calculate_coefficients(self, max_attempts: int = 10) -> Dict[str, np.ndarray]:
        """Calculate coefficients with improved numerical stability"""
        # Try PDF solution first for efficiency
        pdf_works, pdf_residual = self._check_pdf_solution()
        if pdf_works:
            warnings.warn(f"Using known solution from PDF (residual: {pdf_residual})")
            return self.get_all_coefficients()

        def system_equations(vars: np.ndarray) -> np.ndarray:
            self._update_coefficients(vars)
            conditions = _compute_matrix_conditions(
                self.alpha, self.gamma1, self.gamma2,
                self.A0, self.A1, self.B10, self.B31
            )
            if self.det_order == 3:
                det_conditions = _compute_deterministic_conditions(self.alpha, self.A0)
                return np.concatenate([conditions, det_conditions])
            return conditions

        # Try multiple attempts with different initial guesses
        best_residual = np.inf
        best_solution = None
        methods = ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson']

        for attempt in range(max_attempts):
            initial_guess = self._generate_initial_guess()

            # Add small random perturbation after first attempt
            if attempt > 0:
                initial_guess += np.random.normal(0, 0.01, size=len(initial_guess))

            for method in methods:
                try:
                    solution = root(system_equations, initial_guess, method=method, tol=1e-5)
                    if solution.success:
                        residual = np.max(np.abs(solution.fun))
                        if residual < best_residual:
                            best_residual = residual
                            best_solution = solution
                            if residual < 1e-8:  # Good enough solution found
                                self._update_coefficients(best_solution.x)
                                return self.get_all_coefficients()
                except Exception as e:
                    continue  # Try next method if current one fails

        # If no perfect solution found, use the best one if it's reasonable
        if best_solution is not None and best_residual < 1e-2:
            warnings.warn(f"Found solution with residual {best_residual}")
            self._update_coefficients(best_solution.x)
            return self.get_all_coefficients()

        # If still no solution, try fsolve as last resort
        try:
            solution = fsolve(system_equations, self._generate_initial_guess(),
                              full_output=True, xtol=1e-3)
            if solution[2] == 1:  # Check if converged
                self._update_coefficients(solution[0])
                return self.get_all_coefficients()
        except Exception:
            pass

        raise RuntimeError("Failed to find solution after multiple attempts")

    def _update_coefficients(self, vars: np.ndarray) -> None:
        """Update coefficient arrays with Numba optimization"""
        s = self.stages

        # Update vectors
        self.alpha = vars[0:s].astype(np.float64)
        self.gamma1 = vars[s:2 * s].astype(np.float64)
        self.gamma2 = vars[2 * s:3 * s].astype(np.float64)

        # Update matrices efficiently
        idx = 3 * s
        for i in range(1, s):
            for j in range(i):
                self.A0[i, j] = vars[idx]
                self.A1[i, j] = vars[idx + 1]
                self.B10[i, j] = vars[idx + 2]
                self.B31[i, j] = vars[idx + 3]
                idx += 4

        # Update dependent values
        self.c0 = np.sum(self.A0, axis=1)
        self.c1 = np.sum(self.A1, axis=1)

    def _check_pdf_solution(self) -> Tuple[bool, float]:
        """Check if the known PDF solution works"""
        # Define the known solution from the PDF with explicit float values
        pdf_solution = np.array([
            0.25, 0.5, 0.25,  # alpha
            0.707106781, -0.25, 0.25,  # gamma1
            0.0, 0.5, -0.5,  # gamma2
            0.666666667,  # A0[1,0]
            0.666666667, -0.333333333,  # A0[2,0], A0[2,1]
            1.0,  # A1[1,0]
            1.0, 0.0,  # A1[2,0], A1[2,1]
            1.0,  # B10[1,0]
            1.0, 0.0,  # B10[2,0], B10[2,1]
            1.0,  # B31[1,0]
            1.0, 0.0  # B31[2,0], B31[2,1]
        ], dtype=np.float64)

        # Update coefficients with PDF values
        self._update_coefficients(pdf_solution)

        # Check if this solution satisfies our equations
        conditions = _compute_matrix_conditions(
            self.alpha, self.gamma1, self.gamma2,
            self.A0, self.A1, self.B10, self.B31
        )

        if self.det_order == 3:
            det_conditions = _compute_deterministic_conditions(self.alpha, self.A0)
            all_conditions = np.concatenate([conditions, det_conditions])
        else:
            all_conditions = conditions

        residual = np.max(np.abs(all_conditions))
        return residual < 1e-6, residual

    def get_all_coefficients(self) -> Dict[str, np.ndarray]:
        """Return computed coefficients as contiguous arrays"""
        return {
            'alpha': np.ascontiguousarray(self.alpha),
            'gamma1': np.ascontiguousarray(self.gamma1),
            'gamma2': np.ascontiguousarray(self.gamma2),
            'A0': np.ascontiguousarray(self.A0),
            'A1': np.ascontiguousarray(self.A1),
            'B10': np.ascontiguousarray(self.B10),
            'B31': np.ascontiguousarray(self.B31),
            'c(0)': np.ascontiguousarray(self.c0),
            'c(1)': np.ascontiguousarray(self.c1)
        }
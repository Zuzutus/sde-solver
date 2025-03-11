import numpy as np
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path

# Import custom modules
from sde_solver.flexible_config import FlexibleConfig
from sde_solver.vectorized_sde_system import VectorizedSDESystem
from sde_solver.scheme_coefficients import SRKCoefficients
from sde_solver.vectorized_solver import VectorizedAdaptiveSDESolver

def heston_grid_simulation(
        num_sim: int,  # Number of simulations
        S0_range: tuple,  # Range of initial stock prices (min, max)
        v0_range: tuple,  # Range of initial volatilities (min, max)
        time_to_maturity: float,  # Time to maturity in years
        risk_free_rate: float,  # Risk-free rate (annual)
        kappa: float,  # Mean reversion speed
        theta: float,  # Long-term volatility
        xi: float,  # Volatility of volatility
        rho: float,  # Correlation between price and volatility
        dt: float,  # Time step (in years)
        Nsaved: int,  # Number of points to save
        output_dir: str = "results_heston"
):
    """
    Run multiple simulations of the Heston model across a grid of initial conditions.
    
    Parameters:
    -----------
    num_sim : int
        Number of simulations to run
    S0_range : tuple
        Min and max initial stock prices
    v0_range : tuple
        Min and max initial volatilities
    time_to_maturity : float
        Time to maturity in years
    risk_free_rate : float
        Risk-free rate (annual)
    kappa : float
        Mean reversion speed of volatility
    theta : float
        Long-term volatility
    xi : float
        Volatility of volatility
    rho : float
        Correlation between price and volatility
    dt : float
        Time step size (in years)
    Nsaved : int
        Number of points to save per simulation
    output_dir : str
        Directory to save results
    
    Returns:
    --------
    dict
        Dictionary containing all result matrices
    """
    start_time = time.time()

    # Calculate number of steps
    n_steps = int(np.ceil(time_to_maturity / dt))
    t_span = (0.0, time_to_maturity)
    record_every = max(1, int(np.floor(n_steps / Nsaved)))

    # Calculate actual number of saved points
    actual_saved = 1 + (n_steps // record_every)

    print(f"Simulation setup: n_steps={n_steps}, record_every={record_every}, points saved={actual_saved}")

    # Determine grid size - we'll take sqrt(num_sim) points in each dimension
    grid_size = int(np.sqrt(num_sim))
    actual_num_sim = grid_size * grid_size  # This might be slightly different from num_sim

    # Initialize result matrices - each column represents a different simulation
    S = np.zeros((actual_saved, actual_num_sim))  # Stock price
    v = np.zeros((actual_saved, actual_num_sim))  # Variance (volatility squared)
    ln_S = np.zeros((actual_saved, actual_num_sim))  # Log stock price
    
    # Generate starting points on a grid
    S0_values = np.linspace(S0_range[0], S0_range[1], grid_size)
    v0_values = np.linspace(v0_range[0], v0_range[1], grid_size)
    
    # Create mesh grid of starting points
    S0_grid, v0_grid = np.meshgrid(S0_values, v0_values)
    S0_flat = S0_grid.flatten()
    v0_flat = v0_grid.flatten()
    
    starting_points = [(S0_flat[i], v0_flat[i]) for i in range(actual_num_sim)]

    # Print setup information
    print(f"Heston Grid Simulation:")
    print(f"- Risk-free rate: {risk_free_rate}")
    print(f"- Kappa (mean reversion): {kappa}")
    print(f"- Theta (long-term variance): {theta}")
    print(f"- Xi (volatility of volatility): {xi}")
    print(f"- Rho (correlation): {rho}")
    print(f"- Time to maturity: {time_to_maturity} years")
    print(f"- Initial stock price range: {S0_range}")
    print(f"- Initial volatility range: {v0_range}")
    print(f"- Number of simulations: {actual_num_sim} ({grid_size}x{grid_size} grid)")

    # Define drift and diffusion functions for the Heston model
    # We use log-price for the stock price to reduce numerical errors
    
    def drift1(state, t, constants=None):
        """Drift function for log stock price"""
        log_S, v_t = state
        r = constants[0]  # risk-free rate
        
        # Drift term for log(S) is (r - 0.5*v)
        return r - 0.5 * v_t

    def drift2(state, t, constants=None):
        """Drift function for variance"""
        log_S, v_t = state
        k = constants[1]  # kappa
        th = constants[2]  # theta
        
        # Ensure variance is non-negative
        v_t_pos = max(v_t, 1e-10)
        
        # Mean reversion term
        return k * (th - v_t_pos)

    def diffusion1(state, t, constants=None):
        """Diffusion function for log stock price"""
        log_S, v_t = state
        rho_val = constants[4]  # correlation
        
        # Ensure variance is non-negative
        v_t_pos = max(v_t, 1e-10)
        
        # Volatility term
        return np.sqrt(v_t_pos)

    def diffusion2(state, t, constants=None):
        """Diffusion function for variance"""
        log_S, v_t = state
        xi_val = constants[3]  # volatility of volatility
        rho_val = constants[4]  # correlation
        
        # Ensure variance is non-negative
        v_t_pos = max(v_t, 1e-10)
        
        # Volatility of volatility term
        return xi_val * np.sqrt(v_t_pos)

    # Create directory for results if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create scheme coefficients once (outside the loop)
    print("Calculating scheme coefficients...")
    srk = SRKCoefficients(det_order=3, stoch_order=2, stages=3)
    coefficients = srk.calculate_coefficients()
    print("Coefficients calculated successfully")

    # Run simulations for each starting point
    for sim_idx, (S0, v0) in enumerate(starting_points):
        if sim_idx >= actual_num_sim:
            break

        print(f"Starting simulation {sim_idx + 1}/{actual_num_sim} with S0 = {S0:.2f}, v0 = {v0:.4f}")

        # Create configuration
        config = FlexibleConfig(
            r=risk_free_rate,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho
        )

        # Initial state [log(S0), v0]
        y0 = np.array([np.log(S0), v0])

        # Create SDE system for the Heston model
        sde_system = VectorizedSDESystem.from_functions(
            drift_funcs=[drift1, drift2],
            diffusion_funcs=[diffusion1, diffusion2],
            constants=config.to_array()
        )

        # Create solver
        solver = VectorizedAdaptiveSDESolver(sde_system, coefficients)

        # Run the simulation
        t, y = solver.solve(
            t_span=t_span,
            y0=y0,
            n_steps=n_steps,
            dt=dt,
            record_every=record_every
        )

        # Save the results
        ln_S[:len(t), sim_idx] = y[:, 0]
        v[:len(t), sim_idx] = y[:, 1]
        
        # Convert log price to actual price
        S[:len(t), sim_idx] = np.exp(y[:, 0])

    # Save results to files
    np.save(os.path.join(output_dir, "S.npy"), S)
    np.save(os.path.join(output_dir, "v.npy"), v)
    np.save(os.path.join(output_dir, "ln_S.npy"), ln_S)
    np.save(os.path.join(output_dir, "time.npy"), t)
    np.save(os.path.join(output_dir, "S0_values.npy"), S0_values)
    np.save(os.path.join(output_dir, "v0_values.npy"), v0_values)

    # Save metadata
    metadata = {
        "num_sim": actual_num_sim,
        "grid_size": grid_size,
        "S0_range": S0_range,
        "v0_range": v0_range,
        "time_to_maturity": time_to_maturity,
        "risk_free_rate": risk_free_rate,
        "kappa": kappa,
        "theta": theta,
        "xi": xi,
        "rho": rho,
        "dt": dt,
        "n_steps": n_steps,
        "Nsaved": Nsaved
    }
    np.save(os.path.join(output_dir, "metadata.npy"), metadata)

    end_time = time.time()
    print(f"All simulations completed in {end_time - start_time:.2f} seconds.")
    print(f"Results saved to {output_dir}")

    # Return all the matrices in a dictionary
    return {
        "S": S,
        "v": v,
        "ln_S": ln_S,
        "t": t,
        "S0_values": S0_values,
        "v0_values": v0_values,
        "metadata": metadata
    }


def plot_heston_results(results, output_dir="results_heston"):
    """
    Plot results from the Heston model simulation.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from heston_grid_simulation
    output_dir : str
        Directory to save plots
    """
    S = results["S"]
    v = results["v"]
    ln_S = results["ln_S"]
    t = results["t"]
    S0_values = results["S0_values"]
    v0_values = results["v0_values"]
    metadata = results["metadata"]
    
    grid_size = metadata["grid_size"]

    # Create directory for plots if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Plot stock price and volatility paths for a sample of simulations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    num_to_plot = min(10, S.shape[1])  # Plot up to 10 paths
    indices = np.linspace(0, S.shape[1]-1, num_to_plot, dtype=int)
    
    for i, sim_idx in enumerate(indices):
        S0 = S[0, sim_idx]
        v0 = v[0, sim_idx]
        
        ax1.plot(t, S[:, sim_idx], label=f'S0={S0:.2f}, v0={v0:.4f}')
        ax2.plot(t, v[:, sim_idx])
    
    ax1.set_ylabel('Stock Price')
    ax1.set_title('Heston Model Stock Price Paths')
    ax1.legend(loc='upper left', fontsize='small')
    ax1.grid(True)
    
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Variance (v)')
    ax2.set_title('Variance Paths')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heston_paths.png"), dpi=300)

    # Plot terminal price distribution
    plt.figure(figsize=(12, 8))
    plt.hist(S[-1, :], bins=30, alpha=0.7)
    plt.xlabel('Terminal Stock Price')
    plt.ylabel('Frequency')
    plt.title(f'Terminal Price Distribution (T = {metadata["time_to_maturity"]})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "terminal_price_dist.png"), dpi=300)

    # Plot terminal variance distribution
    plt.figure(figsize=(12, 8))
    plt.hist(v[-1, :], bins=30, alpha=0.7)
    plt.xlabel('Terminal Variance')
    plt.ylabel('Frequency')
    plt.title(f'Terminal Variance Distribution (T = {metadata["time_to_maturity"]})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "terminal_variance_dist.png"), dpi=300)

    # Plot 3D surface of terminal prices vs initial conditions
    from mpl_toolkits.mplot3d import Axes3D
    
    # Reshape terminal values back to grid
    S_terminal = S[-1, :].reshape(grid_size, grid_size)
    v_terminal = v[-1, :].reshape(grid_size, grid_size)
    
    # Create meshgrid for 3D plotting
    S0_grid, v0_grid = np.meshgrid(S0_values, v0_values)
    
    # Plot terminal stock price surface
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(S0_grid, v0_grid, S_terminal, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('Initial Price (S0)')
    ax.set_ylabel('Initial Variance (v0)')
    ax.set_zlabel('Terminal Stock Price')
    ax.set_title('Terminal Stock Price by Initial Conditions')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig(os.path.join(output_dir, "terminal_price_surface.png"), dpi=300)
    
    # Plot terminal variance surface
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(S0_grid, v0_grid, v_terminal, cmap='plasma', alpha=0.8)
    
    ax.set_xlabel('Initial Price (S0)')
    ax.set_ylabel('Initial Variance (v0)')
    ax.set_zlabel('Terminal Variance')
    ax.set_title('Terminal Variance by Initial Conditions')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig(os.path.join(output_dir, "terminal_variance_surface.png"), dpi=300)
    
    # Plot volatility smile/skew
    plt.figure(figsize=(12, 8))
    
    # Calculate implied volatility (sqrt of variance) at terminal time
    implied_vol = np.sqrt(v[-1, :])
    
    # Get the initial stock prices for each simulation
    initial_prices = S[0, :]
    
    plt.scatter(initial_prices, implied_vol, alpha=0.5)
    plt.xlabel('Initial Stock Price (S0)')
    plt.ylabel('Terminal Implied Volatility')
    plt.title('Terminal Implied Volatility vs Initial Stock Price (Volatility Smile/Skew)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "volatility_smile.png"), dpi=300)

    plt.close('all')
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    # Example usage with realistic parameters
    results = heston_grid_simulation(
        num_sim=25,  # Will create a 5x5 grid
        S0_range=(80, 120),
        v0_range=(0.01, 0.04),  # Initial variance range (volatility^2)
        time_to_maturity=2*365,
        risk_free_rate=0.05,
        kappa=2.0,       # Mean reversion speed
        theta=0.04,      # Long-term variance (volatility^2)
        xi=0.3,          # Volatility of volatility
        rho=-0.7,        # Negative correlation (typical for equity markets)
        dt=0.1,
        Nsaved=2000
    )
    
    plot_heston_results(results)

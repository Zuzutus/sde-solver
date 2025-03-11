import numpy as np
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path

# Import custom modules
from sde_solver.flexible_config import FlexibleConfig
from sde_solver.sde_system import VectorizedSDESystem
from sde_solver.scheme_coefficients import SRKCoefficients
from sde_solver.solver import VectorizedAdaptiveSDESolver


def black_scholes_grid_simulation(
        num_sim: int,  # Number of simulations
        S0_range: tuple,  # Range of initial stock prices (min, max)
        time_to_maturity: float,  # Time to maturity in years
        risk_free_rate: float,  # Risk-free rate (annual)
        volatility: float,  # Volatility
        dt: float = 0.01,  # Time step (in years)
        Nsaved: int = 100,  # Number of points to save
        output_dir: str = "results_bs"
):
    """
    Run multiple simulations of the Black-Scholes model across a grid of initial stock prices.
    
    Parameters:
    -----------
    num_sim : int
        Number of simulations to run
    S0_range : tuple
        Min and max initial stock prices
    time_to_maturity : float
        Time to maturity in years
    risk_free_rate : float
        Risk-free rate (annual)
    volatility : float
        Volatility
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

    # Initialize result matrices - each column represents a different simulation
    S = np.zeros((actual_saved, num_sim))  # Stock price
    ln_S = np.zeros((actual_saved, num_sim))  # Log stock price
    returns = np.zeros((actual_saved, num_sim))  # Returns

    # Generate starting points
    S0_values = np.linspace(S0_range[0], S0_range[1], num_sim)

    # Print setup information
    print(f"Black-Scholes Grid Simulation:")
    print(f"- Risk-free rate: {risk_free_rate}")
    print(f"- Volatility: {volatility}")
    print(f"- Time to maturity: {time_to_maturity} years")
    print(f"- Initial stock price range: {S0_range}")

    # Define drift and diffusion functions for log-price (using Ito's formula)
    # We use log-price for more accurate simulation
    def drift1(state, t, constants=None):
        """Drift function for log stock price"""
        # We're simulating log(S) to reduce error
        r = constants[0]  # risk-free rate
        sigma = constants[1]  # volatility

        # Drift term for log(S) is (r - 0.5*sigma^2)
        return r - 0.5 * sigma * sigma

    def diffusion1(state, t, constants=None):
        """Diffusion function for log stock price"""
        sigma = constants[1]  # volatility
        return sigma

    # Create directory for results if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create scheme coefficients once (outside the loop)
    print("Calculating scheme coefficients...")
    srk = SRKCoefficients(det_order=3, stoch_order=2, stages=3)
    coefficients = srk.calculate_coefficients()
    print("Coefficients calculated successfully")

    # Run simulations for each starting price
    for sim_idx, S0 in enumerate(S0_values):
        if sim_idx >= num_sim:
            break

        print(f"Starting simulation {sim_idx + 1}/{num_sim} with S0 = {S0:.2f}")

        # Create configuration
        config = FlexibleConfig(
            r=risk_free_rate,
            sigma=volatility
        )

        # Initial state [log(S0)]
        y0 = np.array([np.log(S0)])

        # Create SDE system
        sde_system = VectorizedSDESystem.from_functions(
            drift_funcs=[drift1],
            diffusion_funcs=[diffusion1],
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

        # Save the log stock price
        ln_S[:len(t), sim_idx] = y[:, 0]

        # Convert to stock price
        S[:len(t), sim_idx] = np.exp(y[:, 0])

        # Calculate returns (only after first time point)
        if len(t) > 1:
            returns[1:len(t), sim_idx] = np.diff(y[:, 0]) / dt

    # Save results to files
    np.save(os.path.join(output_dir, "S.npy"), S)
    np.save(os.path.join(output_dir, "ln_S.npy"), ln_S)
    np.save(os.path.join(output_dir, "returns.npy"), returns)
    np.save(os.path.join(output_dir, "time.npy"), t)
    np.save(os.path.join(output_dir, "S0_values.npy"), S0_values)

    # Save metadata
    metadata = {
        "num_sim": num_sim,
        "S0_range": S0_range,
        "time_to_maturity": time_to_maturity,
        "risk_free_rate": risk_free_rate,
        "volatility": volatility,
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
        "ln_S": ln_S,
        "returns": returns,
        "t": t,
        "S0_values": S0_values,
        "metadata": metadata
    }


def plot_black_scholes_results(results, output_dir="results_bs"):
    """
    Plot results from the Black-Scholes simulation.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from black_scholes_grid_simulation
    output_dir : str
        Directory to save plots
    """
    S = results["S"]
    ln_S = results["ln_S"]
    returns = results["returns"]
    t = results["t"]
    S0_values = results["S0_values"]
    metadata = results["metadata"]

    # Create directory for plots if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Plot stock price paths
    plt.figure(figsize=(12, 8))
    for sim_idx in range(min(10, S.shape[1])):  # Plot up to 10 paths
        plt.plot(t, S[:, sim_idx], label=f'S0 = {S0_values[sim_idx]:.2f}')

    plt.xlabel('Time (years)')
    plt.ylabel('Stock Price')
    plt.title('Black-Scholes Stock Price Paths')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "stock_paths.png"), dpi=300)

    # Plot terminal price distribution
    plt.figure(figsize=(12, 8))
    plt.hist(S[-1, :], bins=30, alpha=0.7)
    plt.xlabel('Terminal Stock Price')
    plt.ylabel('Frequency')
    plt.title(f'Terminal Price Distribution (T = {metadata["time_to_maturity"]})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "terminal_price_dist.png"), dpi=300)

    # Plot mean and volatility evolution
    plt.figure(figsize=(12, 8))
    mean_price = np.mean(S, axis=1)
    std_price = np.std(S, axis=1)

    plt.plot(t, mean_price, 'b-', label='Mean Price')
    plt.fill_between(t, mean_price - std_price, mean_price + std_price,
                     color='b', alpha=0.2, label='±1 Std Dev')

    # Add analytical solution
    S0_mean = np.mean(S0_values)
    r = metadata["risk_free_rate"]
    sigma = metadata["volatility"]
    analytical_mean = S0_mean * np.exp(r * t)
    analytical_std = S0_mean * np.sqrt(np.exp(sigma ** 2 * t) * np.exp(2 * r * t) - np.exp(2 * r * t))

    plt.plot(t, analytical_mean, 'r--', label='Analytical Mean')
    plt.fill_between(t, analytical_mean - analytical_std, analytical_mean + analytical_std,
                     color='r', alpha=0.1, label='Analytical ±1 Std Dev')

    plt.xlabel('Time (years)')
    plt.ylabel('Stock Price')
    plt.title('Evolution of Mean and Volatility')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "mean_std_evolution.png"), dpi=300)

    # Plot 3D surface of final prices vs initial prices
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(S0_values, t)
    Z = S

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

    ax.set_xlabel('Initial Price (S0)')
    ax.set_ylabel('Time (years)')
    ax.set_zlabel('Stock Price')
    ax.set_title('Stock Price Evolution by Initial Price')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig(os.path.join(output_dir, "price_surface.png"), dpi=300)

    plt.close('all')
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    results = black_scholes_grid_simulation(
        num_sim=10,
        S0_range=(80, 120),
        time_to_maturity=100,
        risk_free_rate=0.05,
        volatility=0.2,
        dt=0.01,
        Nsaved=1000
    )

    plot_black_scholes_results(results)

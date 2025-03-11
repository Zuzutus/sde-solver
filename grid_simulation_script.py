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
from sde_solver.revised_point_generation import generate_starting_points

def multi_simulation_grid(
        num_sim: int,  # Number of simulations
        pic_size: float,  # Size of picture in nm (square)
        v_amp: float,  # Velocity amplitude
        phi_degrees: float,  # Angle of dragging in degrees
        dt: float = 2.0e-8,  # Time step
        Nsaved: int = 10000,  # Number of points to save
        output_dir: str = "results"
):
    """
    Run multiple simulations to cover a grid pattern across a square picture area.
    
    Parameters:
    -----------
    num_sim : int
        Number of simulations to run
    pic_size : float
        Size of picture in nm (square)
    v_amp : float
        Velocity amplitude
    phi_degrees : float
        Angle of dragging in degrees
    dt : float
        Time step size
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

    # Convert angle to radians
    phi = np.radians(phi_degrees)

    # Calculate velocities based on angle and amplitude
    v_x = v_amp * np.cos(phi)
    v_y = v_amp * np.sin(phi)

    # Calculate number of steps to reach the desired distance
    n_steps = int(np.floor(pic_size / (dt * v_amp)))
    t_span = (0.0, n_steps * dt)
    record_every = max(1, int(np.floor(n_steps / Nsaved)))

    # Calculate actual number of saved points
    actual_saved = 1 + (n_steps // record_every)

    print(f"Simulation setup: n_steps={n_steps}, record_every={record_every}, points saved={actual_saved}")

    # Initialize result matrices - each column represents a different simulation
    X = np.zeros((actual_saved, num_sim))
    Y = np.zeros((actual_saved, num_sim))
    V_x = np.zeros((actual_saved, num_sim))
    V_y = np.zeros((actual_saved, num_sim))
    X_sup_cord = np.zeros((actual_saved, num_sim))
    Y_sup_cord = np.zeros((actual_saved, num_sim))
    FFF_x = np.zeros((actual_saved, num_sim))
    FFF_y = np.zeros((actual_saved, num_sim))
    FFF_R = np.zeros((actual_saved, num_sim))

    # Generate starting points using the revised formula
    grid_positions = generate_starting_points(num_sim, pic_size, phi_degrees)

    # Print setup information
    print(f"Simulation setup: n_steps={n_steps}, record_every={record_every}, points saved={actual_saved}")

    # Define drift and diffusion functions
    def drift1(state, t, constants=None):
        """Drift function for x position (equals x velocity)"""
        return state[1]  # x2 (x velocity)

    def drift2(state, t, constants=None):
        """Drift function for x velocity"""
        # Get constants
        const = constants
        k = const[0]
        m = const[1]
        a = const[2]
        eta = const[5]
        v_x = const[6]
        c_x = const[8]  # Starting position X
        mu = 2.0 * np.sqrt(k / m)

        # Unpack state variables
        x1, x2, y1, _ = state

        # Calculate u0 (potential strength)
        u0 = eta * (a * a) * k / (4.0 * np.pi * np.pi)

        # Calculate terms
        term1 = -mu * x2  # Damping
        term2 = (4.0 * np.pi * u0 / (m * a)) * np.sin(2.0 * np.pi * x1 / a) * \
                np.cos(2.0 * np.pi * y1 / (a * np.sqrt(3.0)))  # Periodic potential
        term3 = (k / m) * (t * v_x + c_x - x1)  # Spring force with starting position

        return term1 + term2 + term3

    def drift3(state, t, constants=None):
        """Drift function for y position (equals y velocity)"""
        return state[3]  # y2 (y velocity)

    def drift4(state, t, constants=None):
        """Drift function for y velocity"""
        # Get constants
        const = constants
        k = const[0]
        m = const[1]
        a = const[2]
        eta = const[5]
        v_y = const[7]
        c_y = const[9]  # Starting position Y
        mu = 2.0 * np.sqrt(k / m)

        # Unpack state variables
        x1, _, y1, y2 = state

        # Calculate u0 (potential strength)
        u0 = eta * (a * a) * k / (4.0 * np.pi * np.pi)
        sqrt3 = np.sqrt(3.0)

        # Calculate terms
        term1 = -mu * y2  # Damping
        term2 = (4.0 * np.pi * u0 / (m * a * sqrt3)) * (
                np.sin(4.0 * np.pi * y1 / (a * sqrt3)) +
                np.sin(2.0 * np.pi * y1 / (a * sqrt3)) * np.cos(2.0 * np.pi * x1 / a)
        )  # Periodic potential
        term3 = (k / m) * (t * v_y + c_y - y1)  # Spring force with starting position

        return term1 + term2 + term3

    def diffusion2(state, t, constants=None):
        """Diffusion function for x velocity (thermal noise)"""
        const = constants
        k = const[0]
        m = const[1]
        kb = const[4]
        T = const[3]
        mu = 2.0 * np.sqrt(k / m)

        return np.sqrt(2.0 * kb * T * mu / m)

    def diffusion4(state, t, constants=None):
        """Diffusion function for y velocity (thermal noise)"""
        const = constants
        k = const[0]
        m = const[1]
        kb = const[4]
        T = const[3]
        mu = 2.0 * np.sqrt(k / m)
        return np.sqrt(2.0 * kb * T * mu / m)

    # Create directory for results if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)


    # Create scheme coefficients once (outside the loop)
    print("Calculating scheme coefficients...")
    srk = SRKCoefficients(det_order=3, stoch_order=2, stages=3)
    coefficients = srk.calculate_coefficients()
    print("Coefficients calculated successfully")

    # Run simulations for each grid position
    for sim_idx, (c_x, c_y) in enumerate(grid_positions):
        if sim_idx >= num_sim:
            break

        print(f"Starting simulation {sim_idx + 1}/{num_sim} at position ({c_x:.4f}, {c_y:.4f})")

        # Create configuration with this starting position
        config = FlexibleConfig(
            k=1.0, m=1.0e-12, a=0.564, T=300, kb=1.3807e-5, eta=20,
            v_x=v_x, v_y=v_y, c_x=c_x, c_y=c_y
        )

        # Initial state [position, velocity] starting at the specified grid position
        y0 = np.array([c_x, 0.0, c_y, 0.0])

        # Create SDE system
        sde_system = VectorizedSDESystem.from_functions(
            drift_funcs=[drift1, drift2, drift3, drift4],
            diffusion_funcs=[None, diffusion2, None, diffusion4],
            constants=config.to_array()
        )

        # Create scheme coefficients
        #srk = SRKCoefficients(det_order=3, stoch_order=2, stages=3)
        #coefficients = srk.calculate_coefficients()

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

        # Save the trajectory results
        X[:len(t), sim_idx] = y[:, 0]
        Y[:len(t), sim_idx] = y[:, 2]
        V_x[:len(t), sim_idx] = y[:, 1]
        V_y[:len(t), sim_idx] = y[:, 3]

        # Create the dragging coordinates (time * velocity + starting position)
        X_sup_cord[:len(t), sim_idx] = t * v_x + c_x
        Y_sup_cord[:len(t), sim_idx] = t * v_y + c_y

        # Calculate forces
        k = 1  # Spring constant
        FFF_x[:len(t), sim_idx] = k * (X_sup_cord[:len(t), sim_idx] - X[:len(t), sim_idx])
        FFF_y[:len(t), sim_idx] = k * (Y_sup_cord[:len(t), sim_idx] - Y[:len(t), sim_idx])

        # Calculate resultant force along dragging direction
        FFF_R[:len(t), sim_idx] = FFF_x[:len(t), sim_idx] * np.cos(phi) + FFF_y[:len(t), sim_idx] * np.sin(phi)

    # Save results to files
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "Y.npy"), Y)
    np.save(os.path.join(output_dir, "V_x.npy"), V_x)
    np.save(os.path.join(output_dir, "V_y.npy"), V_y)
    np.save(os.path.join(output_dir, "X_sup_cord.npy"), X_sup_cord)
    np.save(os.path.join(output_dir, "Y_sup_cord.npy"), Y_sup_cord)
    np.save(os.path.join(output_dir, "FFF_x.npy"), FFF_x)
    np.save(os.path.join(output_dir, "FFF_y.npy"), FFF_y)
    np.save(os.path.join(output_dir, "FFF_R.npy"), FFF_R)
    np.save(os.path.join(output_dir, "time.npy"), t)

    # Save metadata
    metadata = {
        "num_sim": num_sim,
        "pic_size": pic_size,
        "v_amp": v_amp,
        "phi_degrees": phi_degrees,
        "dt": dt,
        "n_steps": n_steps,
        "Nsaved": Nsaved,
        "grid_positions": grid_positions,
        "v_x": v_x,
        "v_y": v_y
    }
    np.save(os.path.join(output_dir, "metadata.npy"), metadata)

    end_time = time.time()
    print(f"All simulations completed in {end_time - start_time:.2f} seconds.")
    print(f"Results saved to {output_dir}")

    # Return all the matrices in a dictionary
    return {
        "X": X,
        "Y": Y,
        "V_x": V_x,
        "V_y": V_y,
        "X_sup_cord": X_sup_cord,
        "Y_sup_cord": Y_sup_cord,
        "FFF_x": FFF_x,
        "FFF_y": FFF_y,
        "FFF_R": FFF_R,
        "t": t,
        "metadata": metadata
    }


def plot_simulation_results(results, sim_idx=0, output_dir="results"):
    """
    Plot results from a specific simulation.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from multi_simulation_grid
    sim_idx : int
        Index of the simulation to plot
    output_dir : str
        Directory to save plots
    """
    X = results["X"]
    Y = results["Y"]
    V_x = results["V_x"]
    V_y = results["V_y"]
    X_sup_cord = results["X_sup_cord"]
    Y_sup_cord = results["Y_sup_cord"]
    FFF_x = results["FFF_x"]
    FFF_y = results["FFF_y"]
    FFF_R = results["FFF_R"]
    t = results["t"]
    metadata = results["metadata"]

    # Create directory for plots if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Plot trajectory
    plt.figure(figsize=(10, 8))
    plt.plot(X[:, sim_idx], Y[:, sim_idx], 'b-', label='Particle')
    plt.plot(X_sup_cord[:, sim_idx], Y_sup_cord[:, sim_idx], 'r--', label='Dragging')
    plt.scatter(X[0, sim_idx], Y[0, sim_idx], c='g', s=50, label='Start')
    plt.scatter(X[-1, sim_idx], Y[-1, sim_idx], c='m', s=50, label='End')
    plt.xlabel('X Position (nm)')
    plt.ylabel('Y Position (nm)')
    plt.title(f'Trajectory (Sim {sim_idx + 1})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"trajectory_sim{sim_idx + 1}.png"), dpi=300)

    # Plot forces vs. dragging distance
    plt.figure(figsize=(10, 8))
    drag_distance = np.sqrt((X_sup_cord[:, sim_idx] - X_sup_cord[0, sim_idx]) ** 2 +
                            (Y_sup_cord[:, sim_idx] - Y_sup_cord[0, sim_idx]) ** 2)
    plt.plot(drag_distance, FFF_x[:, sim_idx], 'b-', label='Force X')
    plt.plot(drag_distance, FFF_y[:, sim_idx], 'r-', label='Force Y')
    plt.plot(drag_distance, FFF_R[:, sim_idx], 'g-', label='Force R')
    plt.xlabel('Dragging Distance (nm)')
    plt.ylabel('Force (pN)')
    plt.title(f'Forces vs. Dragging Distance (Sim {sim_idx + 1})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"forces_sim{sim_idx + 1}.png"), dpi=300)

    # Plot positions and velocities over time
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # X position
    axs[0, 0].plot(t, X[:, sim_idx], 'b-', label='X Position')
    axs[0, 0].plot(t, X_sup_cord[:, sim_idx], 'r--', label='X Dragging')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('X Position (nm)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Y position
    axs[0, 1].plot(t, Y[:, sim_idx], 'b-', label='Y Position')
    axs[0, 1].plot(t, Y_sup_cord[:, sim_idx], 'r--', label='Y Dragging')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Y Position (nm)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # X velocity
    axs[1, 0].plot(t, V_x[:, sim_idx], 'g-')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('X Velocity (nm/s)')
    axs[1, 0].grid(True)

    # Y velocity
    axs[1, 1].plot(t, V_y[:, sim_idx], 'g-')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Y Velocity (nm/s)')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"time_series_sim{sim_idx + 1}.png"), dpi=300)

    plt.close('all')
    print(f"Plots for simulation {sim_idx + 1} saved to {output_dir}")


if __name__ == "__main__":
    # Example usage - these values can be modified
    num_sim = 16  # Number of simulations
    pic_size = 6.0  # Picture size in nm
    v_amp = 1000.0  # Velocity amplitude
    phi_degrees = 30.0  # Angle of dragging in degrees

    # Run the simulations
    results = multi_simulation_grid(
        num_sim=num_sim,
        pic_size=pic_size,
        v_amp=v_amp,
        phi_degrees=phi_degrees
    )

    # Plot results for the first simulation
    plot_simulation_results(results, sim_idx=0)

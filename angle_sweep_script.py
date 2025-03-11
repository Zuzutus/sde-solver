import numpy as np
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path

# Import the simulation function
from grid_simulation_script import multi_simulation_grid

def run_angle_sweep(
    angles_degrees: list,  # List of angles to sweep through
    num_sim: int,
    pic_size: float,
    v_amp: float,
    dt: float = 2.0e-9,
    Nsaved: int = 10000,
    base_output_dir: str = "angle_sweep_results"
):
    """
    Run simulations at multiple dragging angles.
    
    Parameters:
    -----------
    angles_degrees : list
        List of angles in degrees to sweep through
    num_sim : int
        Number of simulations per angle
    pic_size : float
        Size of picture in nm
    v_amp : float
        Velocity amplitude
    dt : float
        Time step size
    Nsaved : int
        Number of points to save per simulation
    base_output_dir : str
        Base directory to save results
    
    Returns:
    --------
    dict
        Dictionary with angle as key and results dictionary as value
    """
    start_time = time.time()
    
    # Create base directory if it doesn't exist
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save sweep parameters
    sweep_params = {
        "angles_degrees": angles_degrees,
        "num_sim": num_sim,
        "pic_size": pic_size,
        "v_amp": v_amp,
        "dt": dt,
        "Nsaved": Nsaved
    }
    np.save(os.path.join(base_output_dir, "sweep_params.npy"), sweep_params)
    
    # Initialize results dictionary
    all_results = {}
    
    # Loop through angles
    for angle in angles_degrees:
        angle_start_time = time.time()
        print(f"\n=== Running simulations at angle {angle} degrees ===\n")
        
        # Create angle-specific output directory
        angle_dir = os.path.join(base_output_dir, f"angle_{angle}")
        
        # Run simulations for this angle
        results = multi_simulation_grid(
            num_sim=num_sim,
            pic_size=pic_size,
            v_amp=v_amp,
            phi_degrees=angle,
            dt=dt,
            Nsaved=Nsaved,
            output_dir=angle_dir
        )
        
        # Store results
        all_results[angle] = results
        
        angle_end_time = time.time()
        print(f"Completed angle {angle}° in {angle_end_time - angle_start_time:.2f} seconds")
        
        # Create visualizations for this angle
        generate_angle_visualizations(results, angle, angle_dir)
    
    # Create summary plots comparing all angles
    generate_angle_comparison(all_results, base_output_dir)
    
    end_time = time.time()
    print(f"\nAngle sweep completed in {end_time - start_time:.2f} seconds.")
    print(f"Results saved to {base_output_dir}")
    
    return all_results


def generate_angle_visualizations(results, angle, output_dir):
    """
    Generate visualizations for a specific angle's results.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from multi_simulation_grid
    angle : float
        Angle in degrees
    output_dir : str
        Directory to save visualizations
    """
    # Extract data
    X = results["X"]
    Y = results["Y"]
    FFF_x = results["FFF_x"]
    FFF_y = results["FFF_y"]
    FFF_R = results["FFF_R"]
    X_sup_cord = results["X_sup_cord"]
    Y_sup_cord = results["Y_sup_cord"]
    
    # Calculate dragging distance along the path for each simulation
    drag_distance = np.sqrt(
        (X_sup_cord - X_sup_cord[:1])**2 + 
        (Y_sup_cord - Y_sup_cord[:1])**2
    )
    
    # Plot average force vs. dragging distance
    plt.figure(figsize=(10, 8))
    
    # Calculate average force across all simulations
    avg_force_r = np.mean(FFF_R, axis=1)
    std_force_r = np.std(FFF_R, axis=1)
    
    # Calculate average dragging distance (should be the same for all simulations)
    avg_distance = np.mean(drag_distance, axis=1)
    
    plt.plot(avg_distance, avg_force_r, 'b-', label='Average Force')
    plt.fill_between(avg_distance, avg_force_r - std_force_r, avg_force_r + std_force_r, 
                     color='b', alpha=0.3, label='±1 Std Dev')
    
    plt.xlabel('Dragging Distance (nm)')
    plt.ylabel('Force (pN)')
    plt.title(f'Average Force vs. Dragging Distance at {angle}°')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_force_vs_distance.png"), dpi=300)
    
    # Plot force distribution heatmap
    plt.figure(figsize=(12, 8))
    
    # Create 2D histogram of forces at different positions
    numY = int(np.ceil(np.sqrt(X.shape[1])))  # Number of grid points in Y direction
    
    # Plot overlay of all particle trajectories
    for i in range(min(X.shape[1], 10)):  # Plot up to 10 trajectories
        plt.plot(X[:, i], Y[:, i], alpha=0.5, lw=0.5)
    
    plt.xlabel('X Position (nm)')
    plt.ylabel('Y Position (nm)')
    plt.title(f'Particle Trajectories at {angle}°')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectories.png"), dpi=300)
    
    # Plot all forces on one figure to see distribution
    plt.figure(figsize=(10, 8))
    for sim_idx in range(min(10, X.shape[1])):  # Plot up to 10 simulations
        plt.plot(drag_distance[:, sim_idx], FFF_R[:, sim_idx], alpha=0.5,
                 label=f'Sim {sim_idx+1}' if sim_idx < 5 else None)
    
    plt.xlabel('Dragging Distance (nm)')
    plt.ylabel('Force (pN)')
    plt.title(f'Force vs. Dragging Distance for Multiple Simulations at {angle}°')
    plt.grid(True)
    if X.shape[1] <= 10:
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_forces_vs_distance.png"), dpi=300)
    
    plt.close('all')


def generate_angle_comparison(all_results, base_output_dir):
    """
    Generate plots comparing results across different angles.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary with angle as key and results dictionary as value
    base_output_dir : str
        Base directory to save visualizations
    """
    angles = sorted(all_results.keys())
    
    # Create plot for average force comparison
    plt.figure(figsize=(12, 8))
    
    for angle in angles:
        results = all_results[angle]
        FFF_R = results["FFF_R"]
        
        # Calculate average force across all simulations
        avg_force = np.mean(FFF_R, axis=1)
        
        # Plot average force vs. time step
        plt.plot(avg_force, label=f'{angle}°')
    
    plt.xlabel('Time Step')
    plt.ylabel('Average Force (pN)')
    plt.title('Average Force for Different Dragging Angles')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir, "angle_comparison_force.png"), dpi=300)
    
    # Create plot for max force vs. angle
    plt.figure(figsize=(10, 6))
    
    max_forces = []
    avg_forces = []
    
    for angle in angles:
        results = all_results[angle]
        FFF_R = results["FFF_R"]
        
        # Calculate metrics across all simulations
        max_force = np.max(np.mean(FFF_R, axis=1))
        avg_force = np.mean(FFF_R)
        
        max_forces.append(max_force)
        avg_forces.append(avg_force)
    
    plt.plot(angles, max_forces, 'bo-', label='Max Force')
    plt.plot(angles, avg_forces, 'ro-', label='Average Force')
    
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Force (pN)')
    plt.title('Maximum and Average Forces vs. Dragging Angle')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir, "forces_vs_angle.png"), dpi=300)
    
    # Plot dragging paths for different angles
    plt.figure(figsize=(10, 10))
    
    for angle in angles:
        results = all_results[angle]
        X_sup_cord = results["X_sup_cord"]
        Y_sup_cord = results["Y_sup_cord"]
        
        # Plot first simulation's dragging path
        plt.plot(X_sup_cord[:, 0], Y_sup_cord[:, 0], label=f'{angle}°')
    
    plt.xlabel('X Position (nm)')
    plt.ylabel('Y Position (nm)')
    plt.title('Dragging Paths for Different Angles')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir, "dragging_paths.png"), dpi=300)
    
    plt.close('all')


if __name__ == "__main__":
    # Example usage - these values can be modified
    angles_to_sweep = [0, 15, 30, 45, 60, 75, 90]  # Angles in degrees
    num_sim = 16      # Number of simulations per angle
    pic_size = 6.0    # Picture size in nm
    v_amp = 1000.0    # Velocity amplitude
    
    # Run the angle sweep
    results = run_angle_sweep(
        angles_degrees=angles_to_sweep,
        num_sim=num_sim,
        pic_size=pic_size,
        v_amp=v_amp
    )

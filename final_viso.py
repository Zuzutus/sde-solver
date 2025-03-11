import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import os
from pathlib import Path
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


def advanced_particle_animation(
        results_dir="results",
        output_dir="animations1",
        sim_idx=0,  # Which simulation to animate
        output_format="mp4",  # mp4, gif, or webm
        fps=15,
        dpi=100,
        colormap='viridis',
        show_force=True,  # Show force vectors
        force_scale=0.5,  # Scaling for force vectors
        time_decimals=6,  # Number of decimal places for time display
        num_frames=None,  # Number of frames (None = use all time steps)
        skip_steps=1,  # Skip every n steps to reduce frames
        quality=95,  # Quality for writer (0-100)
        progress_bar=True  # Show progress bar during creation
):
    """
    Create a high-quality animation of the particle motion on the potential energy surface
    with enhanced visualization and reliable video file output.

    Parameters:
    -----------
    results_dir : str
        Directory containing the simulation results
    output_dir : str
        Directory to save the animation
    sim_idx : int
        Index of the simulation to animate
    output_format : str
        Format of the output video ('mp4', 'gif', or 'webm')
    fps : int
        Frames per second in the output video
    dpi : int
        Resolution of the output video
    colormap : str
        Colormap for the potential energy surface
    show_force : bool
        Whether to show force vectors
    force_scale : float
        Scaling factor for force vectors
    time_decimals : int
        Decimal places for time display
    num_frames : int or None
        Number of frames to generate (None = use all time steps)
    skip_steps : int
        Skip every n steps to reduce total frames
    quality : int
        Quality of the output video (0-100)
    progress_bar : bool
        Whether to show a progress bar during animation creation
    """
    # Import tqdm conditionally for progress bar
    if progress_bar:
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            print("tqdm not available, progress bar disabled")
            has_tqdm = False
    else:
        has_tqdm = False

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load the data
    X = np.load(os.path.join(results_dir, "X.npy"))
    Y = np.load(os.path.join(results_dir, "Y.npy"))
    X_sup_cord = np.load(os.path.join(results_dir, "X_sup_cord.npy"))
    Y_sup_cord = np.load(os.path.join(results_dir, "Y_sup_cord.npy"))
    time = np.load(os.path.join(results_dir, "time.npy"))

    # Load force data if requested
    if show_force:
        FFF_x = np.load(os.path.join(results_dir, "FFF_x.npy"))
        FFF_y = np.load(os.path.join(results_dir, "FFF_y.npy"))
        FFF_R = np.load(os.path.join(results_dir, "FFF_R.npy"))

    # Get dimensions
    num_steps, num_sims = X.shape
    print(f"Data shape: {X.shape}, {num_steps} time steps available")

    # Check if the simulation index is valid
    if sim_idx >= num_sims:
        print(f"Simulation index {sim_idx} is out of range. Using simulation 0 instead.")
        sim_idx = 0

    # Select time steps for animation
    if num_frames is None:
        # Use every nth time step
        time_indices = np.arange(0, num_steps, skip_steps)
    else:
        # Select evenly spaced time steps
        time_indices = np.linspace(0, num_steps - 1, num_frames, dtype=int)

    print(f"Creating animation for simulation {sim_idx} with {len(time_indices)} frames")

    # Calculate the potential energy surface
    def calculate_potential_energy(x_range, y_range, resolution=100):
        """Calculate the potential energy surface using the provided formula."""
        # Define constants
        eta = 20
        aa = 0.564
        k = 1
        u0 = eta * (aa ** 2) * k / (4 * np.pi ** 2)

        # Create meshgrid
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        # Calculate potential energy using the formula
        V = u0 * (np.cos(Y * 4 * np.pi / np.sqrt(3) / aa) +
                  2 * np.cos(X * 2 * np.pi / aa) * np.cos(Y * 2 * np.pi / np.sqrt(3) / aa))

        return X, Y, V

    # Determine the range for the potential energy calculation
    x_min = np.min(X[:, sim_idx]) - 0.5
    x_max = np.max(X[:, sim_idx]) + 0.5
    y_min = np.min(Y[:, sim_idx]) - 0.5
    y_max = np.max(Y[:, sim_idx]) + 0.5

    # Calculate the potential energy surface
    X_pot, Y_pot, V_pot = calculate_potential_energy((x_min, x_max), (y_min, y_max), resolution=150)

    # Create the figure for animation
    fig, ax = plt.subplots(figsize=(12, 10))

    # Initialize plot elements
    contour = ax.contourf(X_pot, Y_pot, V_pot, 50, cmap=colormap, alpha=0.8)

    # Create a colorbar for the potential
    cbar = fig.colorbar(contour, ax=ax, pad=0.01)
    cbar.set_label('Potential Energy')

    # Trajectory line (past path)
    particle_line, = ax.plot([], [], 'r-', lw=1.5, alpha=0.7)

    # Current position markers
    particle_point, = ax.plot([], [], 'ro', ms=8, zorder=10)
    support_point, = ax.plot([], [], 'bo', ms=6, zorder=9)

    # Force vector (if enabled)
    if show_force:
        force_arrow = ax.arrow(0, 0, 0, 0, head_width=0.05, head_length=0.1,
                               fc='g', ec='g', lw=1.5, zorder=11)
        force_arrow_collection = [force_arrow]

    # Text displays
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round'))
    force_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=10,
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round'))

    # Set labels and title
    ax.set_xlabel('X Position (nm)', fontsize=12)
    ax.set_ylabel('Y Position (nm)', fontsize=12)
    ax.set_title(f'Particle Motion on Potential Energy Surface (Simulation {sim_idx})', fontsize=14)

    # Create a custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='r', lw=1.5, label='Particle Trajectory'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label='Particle'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=6, label='Support Point')
    ]
    if show_force:
        legend_elements.append(Line2D([0], [0], color='g', lw=1.5, label='Force Vector'))

    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    # Set consistent axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Function to initialize the animation
    def init():
        particle_line.set_data([], [])
        particle_point.set_data([], [])
        support_point.set_data([], [])
        time_text.set_text('')
        force_text.set_text('')

        # Initialize force arrow
        if show_force:
            force_arrow_collection[0].remove()
            force_arrow_collection[0] = ax.arrow(0, 0, 0, 0, head_width=0.05, head_length=0.1,
                                                 fc='g', ec='g', lw=1.5, zorder=11)

        return [particle_line, particle_point, support_point, time_text,
                force_text] + force_arrow_collection if show_force else []

    # Function to update the animation for each frame
    def update(frame_idx):
        i = time_indices[frame_idx]

        # Update trajectory line (show all points up to current time)
        particle_line.set_data(X[:i + 1, sim_idx], Y[:i + 1, sim_idx])

        # Current positions
        particle_x, particle_y = X[i, sim_idx], Y[i, sim_idx]
        support_x, support_y = X_sup_cord[i, sim_idx], Y_sup_cord[i, sim_idx]

        # Update position markers
        particle_point.set_data([particle_x], [particle_y])
        support_point.set_data([support_x], [support_y])

        # Update time text
        time_text.set_text(f'Time: {time[i]:.{time_decimals}f} s')

        # Update force vector and text if enabled
        if show_force:
            force_x, force_y = FFF_x[i, sim_idx], FFF_y[i, sim_idx]
            force_mag = FFF_R[i, sim_idx]

            # Remove old arrow
            force_arrow_collection[0].remove()

            # Scale the arrow for visibility
            scale = force_scale / max(1.0, np.max(np.abs(FFF_R[:, sim_idx])))
            dx, dy = force_x * scale, force_y * scale

            # Create new arrow at current position
            force_arrow_collection[0] = ax.arrow(particle_x, particle_y, dx, dy,
                                                 head_width=0.05, head_length=0.1,
                                                 fc='g', ec='g', lw=1.5, zorder=11)

            # Update force text
            force_text.set_text(f'Force: {force_mag:.2f} pN')

        return [particle_line, particle_point, support_point, time_text,
                force_text] + force_arrow_collection if show_force else []

    # Create the animation with a progress bar if enabled
    frame_seq = range(len(time_indices))
    if has_tqdm:
        frame_seq = tqdm(frame_seq, desc="Creating animation")

    anim = FuncAnimation(fig, update, frames=frame_seq,
                         init_func=init, blit=True, interval=1000 / fps)

    # Save the animation in the requested format
    print(f"Saving animation as {output_format}...")

    output_path = os.path.join(output_dir, f'particle_motion_sim{sim_idx}.{output_format}')

    if output_format == 'gif':
        # Save as GIF
        try:
            anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
            print(f"Animation saved to {output_path}")
        except Exception as e:
            print(f"Error saving GIF: {e}")
            print("Trying alternative format...")
            output_format = 'mp4'
            output_path = os.path.join(output_dir, f'particle_motion_sim{sim_idx}.mp4')

    if output_format in ['mp4', 'webm']:
        # Save as MP4 or WebM using FFmpeg
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='Python'),
                                  bitrate=-1, codec='libx264' if output_format == 'mp4' else 'libvpx')
            anim.save(output_path, writer=writer, dpi=dpi)
            print(f"Animation saved to {output_path}")
        except Exception as e:
            print(f"Error saving video: {e}")
            print("Make sure FFmpeg is installed correctly.")

            # Try using the default writer as a fallback
            try:
                anim.save(output_path, fps=fps, dpi=dpi)
                print(f"Animation saved using default writer to {output_path}")
            except Exception as e2:
                print(f"Final error saving animation: {e2}")
                print("Could not save animation.")

    plt.close()
    return output_path


def calculate_potential_energy(x_range, y_range, resolution=100):
    """
    Calculate the potential energy surface using the provided formula.

    Parameters:
    -----------
    x_range : tuple (x_min, x_max)
        Range of x coordinates
    y_range : tuple (y_min, y_max)
        Range of y coordinates
    resolution : int
        Number of grid points along each dimension

    Returns:
    --------
    X, Y, V : numpy arrays
        Meshgrid of X, Y coordinates and the potential energy V
    """
    # Define constants
    eta = 20
    aa = 0.564
    k = 1
    u0 = eta * (aa ** 2) * k / (4 * np.pi ** 2)

    # Create meshgrid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Calculate potential energy using the formula provided
    V = u0 * (np.cos(Y * 4 * np.pi / np.sqrt(3) / aa) +
              2 * np.cos(X * 2 * np.pi / aa) * np.cos(Y * 2 * np.pi / np.sqrt(3) / aa))

    return X, Y, V


def plot_potential_with_trajectory(
        results_dir="results",
        output_dir="potential_plots1",
        sim_idx=0,  # Which simulation to plot
        plot_type="3d"  # "3d" or "2d"
):
    """
    Create a static plot of the entire particle trajectory on the potential energy surface.

    Parameters:
    -----------
    results_dir : str
        Directory containing the simulation results
    output_dir : str
        Directory to save plots
    sim_idx : int
        Index of the simulation to plot
    plot_type : str
        Type of plot to create ("3d" or "2d")
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load the data
    X = np.load(os.path.join(results_dir, "X.npy"))
    Y = np.load(os.path.join(results_dir, "Y.npy"))

    # Get the range for the potential calculation
    x_min = np.min(X[:, sim_idx]) - 0.5
    x_max = np.max(X[:, sim_idx]) + 0.5
    y_min = np.min(Y[:, sim_idx]) - 0.5
    y_max = np.max(Y[:, sim_idx]) + 0.5

    # Calculate the potential energy surface
    X_pot, Y_pot, V_pot = calculate_potential_energy((x_min, x_max), (y_min, y_max), resolution=200)

    if plot_type == "3d":
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the potential surface
        surf = ax.plot_surface(X_pot, Y_pot, V_pot, cmap='viridis', alpha=0.8, antialiased=True)
        view_angle = (90, -90)
        ax.view_init(elev=view_angle[0], azim=view_angle[1])

        # Plot the particle trajectory
        z_traj = np.zeros(len(X[:, sim_idx]))
        for i in range(len(z_traj)):
            # Find the nearest grid point
            ix = np.abs(X_pot[0] - X[i, sim_idx]).argmin()
            iy = np.abs(Y_pot[:, 0] - Y[i, sim_idx]).argmin()
            z_traj[i] = V_pot[iy, ix]

        ax.plot(X[:, sim_idx], Y[:, sim_idx], z_traj, 'r-', lw=2, label='Particle Trajectory')
        ax.scatter(X[0, sim_idx], Y[0, sim_idx], z_traj[0], c='g', s=50, label='Start')
        ax.scatter(X[-1, sim_idx], Y[-1, sim_idx], z_traj[-1], c='m', s=50, label='End')

        # Add a color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Potential Energy')

        # Set labels and title
        ax.set_xlabel('X Position (nm)')
        ax.set_ylabel('Y Position (nm)')
        ax.set_zlabel('Potential Energy')
        ax.set_title(f'Particle Trajectory on Potential Energy Surface (Simulation {sim_idx})')
        ax.legend()

    else:
        # Create 2D plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the potential as a contour map
        contour = ax.contourf(X_pot, Y_pot, V_pot, 50, cmap='viridis', alpha=0.7)

        # Plot the particle trajectory
        ax.plot(X[:, sim_idx], Y[:, sim_idx], 'r-', lw=2, label='Particle Trajectory')
        ax.scatter(X[0, sim_idx], Y[0, sim_idx], c='g', s=50, label='Start')
        ax.scatter(X[-1, sim_idx], Y[-1, sim_idx], c='m', s=50, label='End')

        # Add a color bar
        fig.colorbar(contour, ax=ax, label='Potential Energy')

        # Set labels and title
        ax.set_xlabel('X Position (nm)')
        ax.set_ylabel('Y Position (nm)')
        ax.set_title(f'Particle Trajectory on Potential Energy Surface (Simulation {sim_idx})')
        ax.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'trajectory_potential_{plot_type}_sim{sim_idx}.png'), dpi=300)
    plt.close()
    print(f"Saved trajectory plot to {output_dir}/trajectory_potential_{plot_type}_sim{sim_idx}.png")

    return fig


def plot_force_surface_3d(
        results_dir="results",
        output_dir="force_plots",
        sim_idx=None,  # If None, plot all simulations
        skip_steps=20,  # Skip steps for better performance
        view_angle=(90, -90),  # Default view angle
        colormap='Greys',  # Default colormap
        surface_alpha=0.8,  # Transparency of the surface
        dpi=300  # Resolution for saved figure
):
    """
    Create a 3D surface plot of the force magnitude experienced by particles.

    Parameters:
    -----------
    results_dir : str
        Directory containing the simulation results
    output_dir : str
        Directory to save the plot
    sim_idx : int or None
        Index of simulation to plot (None = plot all simulations)
    skip_steps : int
        Skip every n steps to reduce data density
    view_angle : tuple (elev, azim)
        Viewing angle for the 3D plot
    colormap : str
        Colormap for the surface plot
    surface_alpha : float
        Transparency of the surface (0-1)
    dpi : int
        Resolution of the saved figure

    Returns:
    --------
    fig : matplotlib figure
        The figure object containing the plot
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import os
    from pathlib import Path
    from mpl_toolkits.mplot3d import Axes3D

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load the data
    X_sup_cord = np.load(os.path.join(results_dir, "X_sup_cord.npy"))
    Y_sup_cord = np.load(os.path.join(results_dir, "Y_sup_cord.npy"))
    FFF_R = np.load(os.path.join(results_dir, "FFF_R.npy"))

    # Get dimensions
    num_steps, num_sims = X_sup_cord.shape

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Determine which simulations to plot
    if sim_idx is None:
        # Plot all simulations
        plot_indices = range(num_sims)
        title_suffix = "all simulations"
        filename_suffix = "all_sims"
    else:
        # Plot specific simulation
        if sim_idx >= num_sims:
            print(f"Simulation index {sim_idx} is out of range. Using simulation 0 instead.")
            sim_idx = 0
        plot_indices = [sim_idx]
        title_suffix = f"simulation {sim_idx}"
        filename_suffix = f"sim{sim_idx}"

    # Apply normalization for color mapping
    norm = plt.Normalize(FFF_R.min(), FFF_R.max())

    # Plot the surface for each simulation

    surf = ax.plot_surface(
        X_sup_cord[::skip_steps, :],
        Y_sup_cord[::skip_steps, :],
        FFF_R[::skip_steps, :],
        cmap=plt.get_cmap(colormap),
        norm=norm,
        alpha=surface_alpha,
        linewidth=1,
        antialiased=True,
        edgecolor='none'
    )

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Force Magnitude (pN)')

    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    # Set labels and title
    ax.set_xlabel('X Position (nm)')
    ax.set_ylabel('Y Position (nm)')
    ax.set_zlabel('Force Magnitude (pN)')
    ax.set_title(f'3D Force Surface Plot ({title_suffix})')

    # Save the figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'force_surface_3d_{filename_suffix}.png')
    plt.savefig(output_path, dpi=dpi)
    print(f"Saved force surface plot to {output_path}")

    return fig


def plot_force_2d(
        results_dir="results",
        output_dir="force_plots",
        sim_idx=0,  # Which simulation to plot
        skip_steps=20,  # Skip steps for better performance
        plot_type="contour",  # "contour" or "heatmap"
        colormap='viridis',  # Default colormap
        contour_levels=20,  # Number of contour levels
        dpi=300  # Resolution for saved figure
):
    """
    Create a 2D visualization of the force magnitude experienced by particles.

    Parameters:
    -----------
    results_dir : str
        Directory containing the simulation results
    output_dir : str
        Directory to save the plot
    sim_idx : int
        Index of simulation to plot
    skip_steps : int
        Skip every n steps to reduce data density
    plot_type : str
        Type of plot - "contour" or "heatmap"
    colormap : str
        Colormap for the plot
    contour_levels : int
        Number of contour levels (for contour plot)
    dpi : int
        Resolution of the saved figure

    Returns:
    --------
    fig : matplotlib figure
        The figure object containing the plot
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import os
    from pathlib import Path
    from scipy.interpolate import griddata

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load the data
    X_sup_cord = np.load(os.path.join(results_dir, "X_sup_cord.npy"))
    Y_sup_cord = np.load(os.path.join(results_dir, "Y_sup_cord.npy"))
    FFF_R = np.load(os.path.join(results_dir, "FFF_R.npy"))

    # Get dimensions and check simulation index
    num_steps, num_sims = X_sup_cord.shape
    if sim_idx >= num_sims:
        print(f"Simulation index {sim_idx} is out of range. Using simulation 0 instead.")
        sim_idx = 0

    # Extract data for the selected simulation and downsample
    x = X_sup_cord[::skip_steps, sim_idx].flatten()
    y = Y_sup_cord[::skip_steps, sim_idx].flatten()
    z = FFF_R[::skip_steps, sim_idx].flatten()

    # Remove any NaN or infinite values
    valid_indices = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[valid_indices]
    y = y[valid_indices]
    z = z[valid_indices]

    # Add small jitter to avoid coplanar points if needed
    if len(x) > 3:  # Need at least 3 points for triangulation
        # Check if points might be coplanar by looking at variation in coordinates
        x_range = np.ptp(x)
        y_range = np.ptp(y)

        if x_range < 1e-10 or y_range < 1e-10:
            print("Points appear to be nearly coplanar, adding small jitter...")
            np.random.seed(42)  # For reproducibility
            x += np.random.normal(0, 0.0001, size=len(x))
            y += np.random.normal(0, 0.0001, size=len(y))

    # Create a regular grid for interpolation
    xi = np.linspace(np.min(x), np.max(x), 200)
    yi = np.linspace(np.min(y), np.max(y), 200)

    # Try interpolation with different methods, falling back to simpler ones if needed
    try:
        # Try cubic interpolation first
        zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')
    except Exception as e:
        print(f"Cubic interpolation failed: {e}")
        try:
            # Fall back to linear interpolation
            print("Trying linear interpolation instead...")
            zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
        except Exception as e:
            print(f"Linear interpolation failed: {e}")
            # Last resort - nearest neighbor interpolation
            print("Falling back to nearest neighbor interpolation...")
            zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='nearest')

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the plot based on the selected type
    if plot_type == "contour":
        # Contour plot with filled contours
        contour = ax.contourf(xi, yi, zi, contour_levels, cmap=colormap)
        # Add contour lines
        ax.contour(xi, yi, zi, contour_levels, colors='k', alpha=0.3, linewidths=0.5)
    else:  # heatmap
        # Pcolormesh for heatmap
        heatmap = ax.pcolormesh(xi, yi, zi, cmap=colormap, shading='auto')
        contour = heatmap  # For colorbar reference

    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Force Magnitude (pN)')

    # Set labels and title
    ax.set_xlabel('X Position (nm)')
    ax.set_ylabel('Y Position (nm)')
    ax.set_title(f'Force Magnitude Map (Simulation {sim_idx})')

    # Add trajectory line (if available)
    try:
        X = np.load(os.path.join(results_dir, "X.npy"))
        Y = np.load(os.path.join(results_dir, "Y.npy"))
        ax.plot(X[:, sim_idx], Y[:, sim_idx], 'r-', lw=1.5, alpha=0.7, label='Particle Trajectory')
        ax.scatter(X[0, sim_idx], Y[0, sim_idx], c='g', s=50, label='Start')
        ax.scatter(X[-1, sim_idx], Y[-1, sim_idx], c='m', s=50, label='End')
        ax.legend()
    except Exception as e:
        print(f"Could not plot trajectory: {e}")

    # Save the figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'force_{plot_type}_sim{sim_idx}.png')
    plt.savefig(output_path, dpi=dpi)
    print(f"Saved force {plot_type} plot to {output_path}")

    return fig


def plot_force_2d_temp(
        results_dir="results",
        output_dir="force_plots",
        sim_idx=0,  # Which simulation to plot
        skip_steps=20,  # Skip steps for better performance
        plot_type="contour",  # "contour" or "heatmap"
        colormap='viridis',  # Default colormap
        contour_levels=20,  # Number of contour levels
        dpi=300  # Resolution for saved figure
):
    """
    Create a 2D visualization of the force magnitude experienced by particles.

    Parameters:
    -----------
    results_dir : str
        Directory containing the simulation results
    output_dir : str
        Directory to save the plot
    sim_idx : int
        Index of simulation to plot
    skip_steps : int
        Skip every n steps to reduce data density
    plot_type : str
        Type of plot - "contour" or "heatmap"
    colormap : str
        Colormap for the plot
    contour_levels : int
        Number of contour levels (for contour plot)
    dpi : int
        Resolution of the saved figure

    Returns:
    --------
    fig : matplotlib figure
        The figure object containing the plot
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import os
    from pathlib import Path
    from scipy.interpolate import griddata

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load the data
    X_sup_cord = np.load(os.path.join(results_dir, "X_sup_cord.npy"))
    Y_sup_cord = np.load(os.path.join(results_dir, "Y_sup_cord.npy"))
    FFF_R = np.load(os.path.join(results_dir, "FFF_R.npy"))

    # Get dimensions and check simulation index
    num_steps, num_sims = X_sup_cord.shape
    if sim_idx >= num_sims:
        print(f"Simulation index {sim_idx} is out of range. Using simulation 0 instead.")
        sim_idx = 0

    # Extract data for the selected simulation and downsample
    x = X_sup_cord[::skip_steps, :]
    y = Y_sup_cord[::skip_steps, :]
    z = FFF_R[::skip_steps, :]

    # Remove any NaN or infinite values
    #valid_indices = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    xi = x[:, :]
    yi = y[:, :]
    zi = z[:, :]


    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the plot based on the selected type
    if plot_type == "contour":
        # Contour plot with filled contours
        contour = ax.contourf(xi, yi, zi, contour_levels, cmap=colormap)
        # Add contour lines
        ax.contour(xi, yi, zi, contour_levels, colors='k', alpha=0.3, linewidths=0.5)
    else:  # heatmap
        # Pcolormesh for heatmap
        heatmap = ax.pcolormesh(xi, yi, zi, cmap=colormap, shading='auto')
        contour = heatmap  # For colorbar reference

    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Force Magnitude (pN)')

    # Set labels and title
    ax.set_xlabel('X Position (nm)')
    ax.set_ylabel('Y Position (nm)')
    ax.set_title(f'Force Magnitude Map (Simulation {sim_idx})')

    # Add trajectory line (if available)
    try:
        X = np.load(os.path.join(results_dir, "X.npy"))
        Y = np.load(os.path.join(results_dir, "Y.npy"))
        ax.plot(X[:, sim_idx], Y[:, sim_idx], 'r-', lw=1.5, alpha=0.7, label='Particle Trajectory')
        ax.scatter(X[0, sim_idx], Y[0, sim_idx], c='g', s=50, label='Start')
        ax.scatter(X[-1, sim_idx], Y[-1, sim_idx], c='m', s=50, label='End')
        ax.legend()
    except Exception as e:
        print(f"Could not plot trajectory: {e}")

    # Save the figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'force_{plot_type}_sim{sim_idx}.png')
    plt.savefig(output_path, dpi=dpi)
    print(f"Saved force {plot_type} plot to {output_path}")

    return fig


if __name__ == "__main__":
    # Example usage - Customize the parameters as needed
    plot_force_surface_3d(
        results_dir="results",
        output_dir="force_plots",
        sim_idx=None,  # If None, plot all simulations
        skip_steps=20,  # Skip steps for better performance
        view_angle=(90, -90),  # Default view angle
        colormap='Greys',  # Default colormap
        surface_alpha=0.8,  # Transparency of the surface
        dpi=300  # Resolution for saved figure
    )

    plot_force_2d_temp(
        results_dir="results",
        output_dir="force_plots",
        sim_idx=0,  # Which simulation to plot
        skip_steps=20,  # Skip steps for better performance
        plot_type="heatmap",  # "contour" or "heatmap"
        colormap='viridis',  # Default colormap
        contour_levels=20,  # Number of contour levels
        dpi=300  # Resolution for saved figure
    )
    # 2. Create a high-quality animation saved as a video file
    #advanced_particle_animation(
    #    results_dir=r"F:\test\sde-solver\results",  # Directory with simulation results
    #    sim_idx=0,              # First simulation
    #    output_format="mp4",    # Save as MP4 video
    #    fps=30,                 # Frames per second
    #    dpi=100,                # Resolution
    #    colormap='viridis',     # Colormap for potential
    #    show_force=True,        # Show force vectors
    #    force_scale=0.5,        # Scale force vectors for visibility
    #    num_frames=30*30,         # Number of frames to generate
    #    progress_bar=True       # Show progress during creation
    #)
    # 3. Create static plots of the trajectory on the potential
    plot_potential_with_trajectory(
        results_dir="results",
        sim_idx=12,  # First simulation
        plot_type="3d"  # "3d" or "2d"
    )

    plot_potential_with_trajectory(
        results_dir="results",
        sim_idx=12,  # First simulation
        plot_type="2d"  # "3d" or "2d"
    )

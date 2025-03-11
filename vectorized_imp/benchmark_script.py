import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Import both original and optimized implementations
from grid_simulation_script import multi_simulation_grid
from grid_simulation_optimized import multi_simulation_grid_optimized


def run_benchmark(num_sims_list, pic_size=6.0, v_amp=1000.0, phi_degrees=30.0, 
                 dt=2.0e-8, Nsaved=1000, use_batch=True):
    """
    Run performance benchmark comparing original vs optimized implementations.
    
    Parameters:
    -----------
    num_sims_list : list
        List of number of simulations to test
    pic_size : float
        Size of picture in nm
    v_amp : float
        Velocity amplitude
    phi_degrees : float
        Angle of dragging in degrees
    dt : float
        Time step size
    Nsaved : int
        Number of points to save per simulation
    use_batch : bool
        Whether to use batch processing in optimized version
        
    Returns:
    --------
    dict
        Dictionary with benchmark results
    """
    results = {
        "num_sims": num_sims_list,
        "original_time": [],
        "optimized_time": [],
        "speedup": [],
        "batch_mode": use_batch
    }
    
    # Create output directory for results
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    for num_sim in num_sims_list:
        print(f"\n===== Testing with {num_sim} simulations =====")
        
        # Run original implementation
        print("\nRunning original implementation...")
        start_time = time.time()
        original_results = multi_simulation_grid(
            num_sim=num_sim,
            pic_size=pic_size,
            v_amp=v_amp,
            phi_degrees=phi_degrees,
            dt=dt,
            Nsaved=Nsaved,
            output_dir=f"benchmark_results/original_{num_sim}"
        )
        original_time = time.time() - start_time
        print(f"Original implementation: {original_time:.2f} seconds")
        
        # Run optimized implementation
        print("\nRunning optimized implementation...")
        start_time = time.time()
        optimized_results = multi_simulation_grid_optimized(
            num_sim=num_sim,
            pic_size=pic_size,
            v_amp=v_amp,
            phi_degrees=phi_degrees,
            dt=dt,
            Nsaved=Nsaved,
            output_dir=f"benchmark_results/optimized_{num_sim}",
            use_batch_processing=use_batch
        )
        optimized_time = time.time() - start_time
        print(f"Optimized implementation: {optimized_time:.2f} seconds")
        
        # Calculate speedup
        speedup = original_time / optimized_time
        print(f"Speedup: {speedup:.2f}x faster")
        
        # Store results
        results["original_time"].append(original_time)
        results["optimized_time"].append(optimized_time)
        results["speedup"].append(speedup)
        
        # Verify results match (at least approximately)
        x_diff = np.abs(original_results["X"] - optimized_results["X"]).max()
        y_diff = np.abs(original_results["Y"] - optimized_results["Y"]).max()
        print(f"Maximum absolute difference - X: {x_diff:.6e}, Y: {y_diff:.6e}")
        
        # Save verification plot for the first simulation
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(original_results["X"][:, 0], original_results["Y"][:, 0], 'b-', 
                label='Original Implementation')
        ax.plot(optimized_results["X"][:, 0], optimized_results["Y"][:, 0], 'r--', 
                label='Optimized Implementation')
        ax.set_xlabel('X Position (nm)')
        ax.set_ylabel('Y Position (nm)')
        ax.set_title(f'Trajectory Comparison (n={num_sim} simulations, first trajectory)')
        ax.legend()
        ax.grid(True)
        fig.savefig(os.path.join(output_dir, f"comparison_{num_sim}.png"), dpi=300)
        plt.close(fig)
    
    # Plot and save benchmark results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(results["num_sims"], results["original_time"], 'b-o', label='Original')
    ax1.plot(results["num_sims"], results["optimized_time"], 'r-o', label='Optimized')
    ax1.set_xlabel('Number of Simulations')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Execution Time Comparison')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(results["num_sims"], results["speedup"], 'g-o')
    ax2.set_xlabel('Number of Simulations')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Performance Speedup')
    ax2.grid(True)
    
    batch_label = "with" if use_batch else "without"
    fig.suptitle(f'Performance Benchmark (Optimized {batch_label} Batch Processing)')
    fig.tight_layout()
    
    fig.savefig(os.path.join(output_dir, "benchmark_results.png"), dpi=300)
    plt.close(fig)
    
    # Save numerical results
    np.save(os.path.join(output_dir, "benchmark_data.npy"), results)
    
    return results


def detailed_profiling(num_sim=16, pic_size=6.0, dt=2.0e-8):
    """
    Run detailed profiling on the most time-consuming parts of the simulation.
    
    Parameters:
    -----------
    num_sim : int
        Number of simulations
    pic_size : float
        Size of picture in nm
    dt : float
        Time step size
    """
    import cProfile
    import pstats
    
    output_dir = Path("benchmark_results/profiling")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Profile original implementation
    cProfile.run(
        'multi_simulation_grid(num_sim=num_sim, pic_size=pic_size, '
        'v_amp=1000.0, phi_degrees=30.0, dt=dt, Nsaved=1000, '
        'output_dir="benchmark_results/profiling/original")',
        'benchmark_results/profiling/original.prof'
    )
    
    # Profile optimized without batch
    cProfile.run(
        'multi_simulation_grid_optimized(num_sim=num_sim, pic_size=pic_size, '
        'v_amp=1000.0, phi_degrees=30.0, dt=dt, Nsaved=1000, '
        'output_dir="benchmark_results/profiling/optimized_nobatch", '
        'use_batch_processing=False)',
        'benchmark_results/profiling/optimized_nobatch.prof'
    )
    
    # Profile optimized with batch
    cProfile.run(
        'multi_simulation_grid_optimized(num_sim=num_sim, pic_size=pic_size, '
        'v_amp=1000.0, phi_degrees=30.0, dt=dt, Nsaved=1000, '
        'output_dir="benchmark_results/profiling/optimized_batch", '
        'use_batch_processing=True)',
        'benchmark_results/profiling/optimized_batch.prof'
    )
    
    # Convert profiling results to text reports
    for name in ['original', 'optimized_nobatch', 'optimized_batch']:
        stats = pstats.Stats(f'benchmark_results/profiling/{name}.prof')
        stats.strip_dirs().sort_stats('cumulative')
        stats.print_stats(30)  # Print top 30 functions
        
        # Save to file
        with open(f'benchmark_results/profiling/{name}_report.txt', 'w') as f:
            stats = pstats.Stats(f'benchmark_results/profiling/{name}.prof', stream=f)
            stats.strip_dirs().sort_stats('cumulative')
            stats.print_stats(50)  # More detailed in file
    
    print("Profiling completed. Results are in benchmark_results/profiling/")


def scaling_test(max_sims=64, step=8):
    """
    Test how the optimized implementation scales with the number of simulations.
    
    Parameters:
    -----------
    max_sims : int
        Maximum number of simulations to test
    step : int
        Step size for number of simulations
    """
    # Test with small simulations for quicker results
    sim_sizes = list(range(step, max_sims+1, step))
    
    # Run with batch processing
    print("\n===== Scaling test with batch processing =====")
    batch_results = run_benchmark(
        num_sims_list=sim_sizes,
        pic_size=6.0,
        v_amp=1000.0,
        phi_degrees=30.0,
        dt=2.0e-8,
        Nsaved=1000,
        use_batch=True
    )
    
    # Run without batch processing
    print("\n===== Scaling test without batch processing =====")
    nobatch_results = run_benchmark(
        num_sims_list=sim_sizes,
        pic_size=6.0,
        v_amp=1000.0,
        phi_degrees=30.0,
        dt=2.0e-8,
        Nsaved=1000,
        use_batch=False
    )
    
    # Plot comparison of batch vs no-batch
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(batch_results["num_sims"], batch_results["optimized_time"], 'g-o', 
             label='With Batch Processing')
    ax1.plot(nobatch_results["num_sims"], nobatch_results["optimized_time"], 'm-o', 
             label='Without Batch Processing')
    ax1.set_xlabel('Number of Simulations')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Execution Time Comparison')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(batch_results["num_sims"], batch_results["speedup"], 'g-o', 
             label='With Batch Processing')
    ax2.plot(nobatch_results["num_sims"], nobatch_results["speedup"], 'm-o', 
             label='Without Batch Processing')
    ax2.set_xlabel('Number of Simulations')
    ax2.set_ylabel('Speedup Factor (vs Original)')
    ax2.set_title('Performance Speedup')
    ax2.legend()
    ax2.grid(True)
    
    fig.suptitle('Scaling Performance: Batch vs Non-Batch Processing')
    fig.tight_layout()
    
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(os.path.join(output_dir, "batch_vs_nobatch.png"), dpi=300)
    
    # Save combined results
    combined_results = {
        "batch": batch_results,
        "nobatch": nobatch_results
    }
    np.save(os.path.join(output_dir, "scaling_test_data.npy"), combined_results)
    
    return combined_results


if __name__ == "__main__":
    # Basic benchmark with small, medium, and large numbers of simulations
    benchmark_results = run_benchmark(
        num_sims_list=[4, 8, 16],
        pic_size=6.0,
        v_amp=1000.0,
        phi_degrees=30.0,
        dt=2.0e-8,
        Nsaved=1000,
        use_batch=True
    )
    
    # Optional: Run detailed profiling (this takes longer)
    # detailed_profiling(num_sim=8)
    
    # Optional: Run scaling test (this takes even longer)
    # scaling_results = scaling_test(max_sims=32, step=8)
    
    print("\nBenchmark complete! Results saved to benchmark_results/")

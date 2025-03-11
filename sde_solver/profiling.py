import cProfile
import pstats
import io
import time
import os
from contextlib import contextmanager


class Profiler:
    """Utility class for profiling code execution."""

    def __init__(self, enabled=True, output_dir="profiling_results"):
        """
        Initialize the profiler.

        Parameters:
        -----------
        enabled : bool
            Whether profiling is enabled
        output_dir : str
            Directory to save profiling results
        """
        self.enabled = enabled
        self.output_dir = output_dir
        if enabled and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.profiler = cProfile.Profile()
        self.start_time = None

    def start(self):
        """Start the profiler."""
        if self.enabled:
            self.start_time = time.time()
            self.profiler.enable()
        return self

    def stop(self, name="profile"):
        """
        Stop the profiler and save results.

        Parameters:
        -----------
        name : str
            Name prefix for output files
        """
        if not self.enabled:
            return None

        self.profiler.disable()
        duration = time.time() - self.start_time

        # Save raw profiling data
        self.profiler.dump_stats(f"{self.output_dir}/{name}.prof")

        # Save formatted text report
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(30)  # Print top 30 functions by cumulative time

        with open(f"{self.output_dir}/{name}_report.txt", 'w') as f:
            f.write(f"Total execution time: {duration:.4f} seconds\n\n")
            f.write(s.getvalue())

        # Return stats object for further analysis
        return ps

    @contextmanager
    def profile_section(self, name):
        """
        Context manager for profiling a section of code.

        Parameters:
        -----------
        name : str
            Name of the section for the output file
        """
        section_profiler = Profiler(enabled=self.enabled, output_dir=self.output_dir)
        section_profiler.start()
        try:
            yield
        finally:
            section_profiler.stop(name=name)

    def print_top_functions(self, n=20):
        """
        Print the top N time-consuming functions.

        Parameters:
        -----------
        n : int
            Number of functions to display
        """
        if not self.enabled:
            return

        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(n)
        print(s.getvalue())


# Simple function-based interface for quick profiling
def profile_function(func):
    """
    Decorator to profile a function.

    Parameters:
    -----------
    func : callable
        Function to profile

    Returns:
    --------
    callable
        Wrapped function that profiles execution
    """

    def wrapper(*args, **kwargs):
        profiler = Profiler()
        profiler.start()
        result = func(*args, **kwargs)
        profiler.stop(name=func.__name__)
        profiler.print_top_functions()
        return result

    return wrapper



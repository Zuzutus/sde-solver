import numpy as np
from typing import Dict, List, Any, Optional


class FlexibleConfig:
    """
    A flexible configuration class that allows any parameter names
    and provides helper methods for accessing parameters in JIT functions.
    """
    
    def __init__(self, **kwargs):
        # Store all parameters in a dictionary
        self.params = kwargs
        # Create ordered list of parameter names (for consistent array mapping)
        self.param_names = list(kwargs.keys())
        # Create index mapping
        self.index_map = {name: i for i, name in enumerate(self.param_names)}
        
    def __getattr__(self, name):
        # Allow attribute-style access
        if name in self.params:
            return self.params[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a numpy array in consistent order"""
        return np.array([self.params[name] for name in self.param_names], dtype=np.float64)
    
    def get_index_map(self) -> Dict[str, int]:
        """Return the mapping of parameter names to array indices"""
        return self.index_map
    
    def print_index_mapping(self):
        """Print the mapping of parameter names to array indices for documentation"""
        print("Parameter to index mapping (for use in JIT functions):")
        for name, idx in self.index_map.items():
            print(f"  {name} = constants[{idx}]")
    
    def generate_constants_comment(self) -> str:
        """Generate a comment block that can be added to JIT functions"""
        comment = "# Parameter index mapping:\n"
        for name, idx in self.index_map.items():
            comment += f"# {name} = constants[{idx}]\n"
        return comment


# Example usage in a script:
if __name__ == "__main__":
    # Create a configuration with arbitrary parameters
    config = FlexibleConfig(G=9.81, mass=2.5, drag=0.1, initial_velocity=20.0)
    
    # Access parameters by name
    print(f"Gravity: {config.G}")
    print(f"Mass: {config.mass}")
    
    # Convert to array for JIT functions
    constants_array = config.to_array()
    print(f"Constants array: {constants_array}")
    
    # Print the index mapping for documentation
    config.print_index_mapping()
    
    # Generate a comment block
    comment = config.generate_constants_comment()
    print("\nSample JIT function with parameter mapping:")
    print(comment)
    print("@jit(nopython=True)")
    print("def compute_acceleration(state, t, constants):")
    print("    # Extract parameters")
    print("    G = constants[0]")
    print("    mass = constants[1]")
    print("    drag = constants[2]")
    print("    # Compute acceleration")
    print("    return G - drag * state[1] / mass")

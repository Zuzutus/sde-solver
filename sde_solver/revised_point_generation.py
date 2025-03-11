def generate_starting_points(
    num_sim: int,         # Number of simulations
    pic_size: float,      # Size of picture in nm (square)
    phi_degrees: float    # Angle of dragging in degrees
):
    """
    Generate starting points for simulations based on specified pattern.
    
    Parameters:
    -----------
    num_sim : int
        Number of simulations to run
    pic_size : float
        Size of picture in nm (square)
    phi_degrees : float
        Angle of dragging in degrees
    
    Returns:
    --------
    list
        List of (x, y) tuples representing starting positions
    """
    import numpy as np
    
    # Convert angle to radians
    phi = np.radians(phi_degrees)
    
    # Calculate starting_X and ending_Y based on phi
    if phi_degrees != 0:
        starting_X = np.sin(phi) * pic_size
        ending_Y = np.cos(phi) * pic_size
    else:
        starting_X = 0
        ending_Y = pic_size
    
    # Calculate dx and dy
    numY = num_sim # Number of points to generate
    if numY <= 1:
        numY = 2  # Ensure at least 2 points
        
    dy = ending_Y / numY
    if phi_degrees != 0:
        dx = starting_X / numY
    else:
        dx = 0
        
    # First point is at (starting_X, ending_Y)
    x0 = starting_X
    y0 = 0
    
    # Generate points using the formula:
    # x = x0 - dx*i
    # y = y0 + dy*i
    starting_points = []
    
    for i in range(numY):
        if len(starting_points) < num_sim:
            # Use the specified formula
            x = x0 - dx * i
            y = y0 + dy * i

            starting_points.append((x, y))
    
    # Print the first few points for verification
    print("Generated starting points:")
    for idx, (x, y) in enumerate(starting_points[:min(5, len(starting_points))]):
        print(f"  Point {idx+1}: ({x:.4f}, {y:.4f})")
    if len(starting_points) > 5:
        print(f"  ... and {len(starting_points)-5} more points")
        
    return starting_points

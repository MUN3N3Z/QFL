import numpy as np
from itertools import product
import math

def read_states_from_file(filename):
    """
    Read states from a file with x,y coordinates.
    
    Args:
        filename (str): Path to the input file
    
    Returns:
        list of tuples: List of (x,y) state coordinates
    """
    states = []
    with open(filename, 'r') as f:
        for line in f:
            line = line[2:-3]
            x, y = map(float, line.split(','))
            states.append((x, y))
    return states

def find_optimal_partition(states, verbose=True):
    """
    Find the most balanced 3x3 partition for the given states.
    
    Args:
        states (list): List of (x,y) state coordinates
        verbose (bool): Whether to print detailed information
    
    Returns:
        dict: Optimal partition boundaries and statistics
    """
    print(len(states))
    indices = np.random.choice(len(states), 100000, replace=False)
    states = [states[i] for i in indices]
    # Extract x and y coordinates
    x_coords = [s[0] for s in states]
    y_coords = [s[1] for s in states]
    
    # Compute percentiles for initial boundary guesses
    x_percentiles = np.percentile(x_coords, [33.33, 66.66])
    y_percentiles = np.percentile(y_coords, [33.33, 66.66])
    
    best_balance_ratio = float('inf')
    best_partition = None
    
    # Try different boundary combinations
    for x1_factor in np.linspace(0.5, 1.5, 10):
        for x2_factor in np.linspace(0.5, 1.5, 10):
            for y1_factor in np.linspace(0.5, 1.5, 10):
                for y2_factor in np.linspace(0.5, 1.5, 10):
                    # Compute boundaries
                    x_bounds = [
                        max(0, x_percentiles[0] * x1_factor), 
                        max(0, x_percentiles[1] * x2_factor)
                    ]
                    y_bounds = [
                        max(0, y_percentiles[0] * y1_factor), 
                        max(0, y_percentiles[1] * y2_factor)
                    ]
                    
                    # Compute partition counts
                    partition_counts = np.zeros((3, 3), dtype=int)
                    
                    for x, y in states:
                        # Determine x partition
                        x_part = 0 if x < x_bounds[0] else (1 if x < x_bounds[1] else 2)
                        
                        # Determine y partition
                        y_part = 0 if y < y_bounds[0] else (1 if y < y_bounds[1] else 2)
                        
                        # Increment partition count
                        partition_counts[x_part, y_part] += 1
                    
                    # Compute balance metrics
                    min_count = np.min(partition_counts)
                    max_count = np.max(partition_counts)
                    
                    # Balance ratio: lower is better
                    balance_ratio = max_count / (min_count + 1e-10)
                    
                    # Update best partition if this is more balanced
                    if balance_ratio < best_balance_ratio:
                        best_balance_ratio = balance_ratio
                        best_partition = {
                            'x_bounds': x_bounds,
                            'y_bounds': y_bounds,
                            'partition_counts': partition_counts,
                            'balance_ratio': balance_ratio
                        }
    
    # Verbose output
    if verbose:
        print("\n--- Optimal 3x3 Partition Results ---")
        print(f"X Boundaries: {best_partition['x_bounds']}")
        print(f"Y Boundaries: {best_partition['y_bounds']}")
        print("\nPartition Distribution:")
        print(best_partition['partition_counts'])
        print(f"\nBalance Ratio: {best_partition['balance_ratio']:.2f}")
    
    return best_partition

def visualize_partition(states, partition):
    """
    Create a scatter plot visualizing the partition
    
    Args:
        states (list): List of (x,y) coordinates
        partition (dict): Partition information
    """
    import matplotlib.pyplot as plt
    
    # Extract coordinates
    x_coords = [s[0] for s in states]
    y_coords = [s[1] for s in states]
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.scatter(x_coords, y_coords, alpha=0.5)
    
    # Draw partition lines
    x_bounds = partition['x_bounds']
    y_bounds = partition['y_bounds']
    
    plt.axvline(x=x_bounds[0], color='r', linestyle='--')
    plt.axvline(x=x_bounds[1], color='r', linestyle='--')
    plt.axhline(y=y_bounds[0], color='g', linestyle='--')
    plt.axhline(y=y_bounds[1], color='g', linestyle='--')
    
    plt.title('State Space Partition')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.tight_layout()
    plt.show()

def main(filename):
    """
    Main function to process states and find optimal partition
    
    Args:
        filename (str): Path to input file with states
    """
    # Read states from file
    states = read_states_from_file(filename)
    
    # Find optimal partition
    partition = find_optimal_partition(states)
    
    # Visualize partition
    try:
        visualize_partition(states, partition)
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file>")
        sys.exit(1)
    
    main(sys.argv[1])
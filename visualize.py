import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from eda_env import EDAFloorplanEnv

def draw_floorplan(env: EDAFloorplanEnv, step_number: int = 0):
    """Generates a professional EDA visualizer for the current environment state."""
    grid = np.array(env.grid)
    size = env.grid_size
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw the base silicon grid
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.set_xticks(np.arange(size))
    ax.set_yticks(np.arange(size))
    ax.grid(which='both', color='lightgrey', linestyle='-', linewidth=1)
    
    # Plot placed components
    cmap = plt.get_cmap("tab10") # Color palette for components
    for comp_id, (x, y) in env.placed_locations.items():
        # Draw component block
        rect = patches.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, 
                               linewidth=2, edgecolor='black', 
                               facecolor=cmap(comp_id % 10), alpha=0.8)
        ax.add_patch(rect)
        # Add label
        ax.text(x, y, f"C{comp_id}", ha='center', va='center', 
                color='white', fontweight='bold', fontsize=12)

    # Draw HPWL Bounding Boxes for Nets
    for i, net in enumerate(env.netlist):
        placed_nodes = [node for node in net if node in env.placed_locations]
        if len(placed_nodes) > 1:
            xs = [env.placed_locations[n][0] for n in placed_nodes]
            ys = [env.placed_locations[n][1] for n in placed_nodes]
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Draw the routing bounding box
            width = max_x - min_x
            height = max_y - min_y
            box = patches.Rectangle((min_x, min_y), width, height, 
                                  linewidth=2.5, edgecolor=cmap(i % 10), 
                                  facecolor='none', linestyle='--')
            ax.add_patch(box)

    plt.title(f"EDA Floorplan State - Task: {env.task_name} | Step: {step_number}", fontsize=14, pad=20)
    plt.gca().invert_yaxis() # Match matrix coordinates (0,0 at top left)
    plt.show()

if __name__ == "__main__":
    # Test the visualizer with a dummy placement
    print("Generating visualizer preview...")
    env = EDAFloorplanEnv(task_name="place_routed")
    
    # Simulate a few steps for the visualizer
    from eda_env import EDAAction
    env.step(EDAAction(component_id=1, x=1, y=1))
    env.step(EDAAction(component_id=2, x=3, y=2))
    env.step(EDAAction(component_id=4, x=2, y=5))
    
    draw_floorplan(env, step_number=3)
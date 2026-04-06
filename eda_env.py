from typing import List, Dict, Any, Tuple
from pydantic import BaseModel

# --- OPENENV SPEC TYPED MODELS ---
class EDAAction(BaseModel):
    component_id: int
    x: int
    y: int

class EDAObservation(BaseModel):
    grid_size: int
    grid_state: List[List[int]]
    unplaced_components: List[int]
    netlist: List[List[int]] # Now supports multi-pin nets e.g., [1, 2, 4]
    congestion_map: List[List[int]] # Tracks routing density (HPWL overlaps)
    message: str

# --- ENVIRONMENT LOGIC ---
class EDAFloorplanEnv:
    def __init__(self, task_name: str = "place_basic"):
        self.task_name = task_name
        self._setup_task()
        self.reset()

    def _setup_task(self):
        """Advanced real-world layout configurations."""
        if self.task_name == "place_basic":
            self.grid_size = 5
            self.total_components = 4
            self.netlist = [[1, 2, 3], [3, 4]] # Multi-terminal nets
            self.thermal_penalty = False
        elif self.task_name == "place_routed":
            self.grid_size = 8
            self.total_components = 6
            self.netlist = [[1, 2, 4], [2, 3, 5], [4, 5, 6]]
            self.thermal_penalty = False
        elif self.task_name == "place_thermal":
            self.grid_size = 10
            self.total_components = 8
            self.netlist = [[1, 2, 3, 4], [3, 5, 6], [2, 7, 8], [6, 8]]
            self.thermal_penalty = True 
        else:
            raise ValueError(f"Unknown task: {self.task_name}")

        self.max_reward_per_step = 1.0 / self.total_components

    def reset(self) -> EDAObservation:
        self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.congestion_map = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.placed_locations = {} 
        self.unplaced_components = list(range(1, self.total_components + 1))
        self.steps_taken = 0
        return self.state()

    def state(self) -> EDAObservation:
        return EDAObservation(
            grid_size=self.grid_size,
            grid_state=self.grid,
            unplaced_components=self.unplaced_components,
            netlist=self.netlist,
            congestion_map=self.congestion_map,
            message=f"Placed {len(self.placed_locations)}/{self.total_components}."
        )

    def step(self, action: EDAAction) -> Tuple[EDAObservation, float, bool, Dict[str, Any]]:
        self.steps_taken += 1
        error_msg = None
        reward_delta = 0.0

        if action.component_id not in self.unplaced_components:
            error_msg = f"invalid_id_{action.component_id}"
        elif not (0 <= action.x < self.grid_size and 0 <= action.y < self.grid_size):
            error_msg = f"out_of_bounds_{action.x}_{action.y}"
        elif self.grid[action.y][action.x] != 0:
            error_msg = f"collision_at_{action.x}_{action.y}"
        else:
            # Valid Placement
            self.grid[action.y][action.x] = action.component_id
            self.placed_locations[action.component_id] = (action.x, action.y)
            self.unplaced_components.remove(action.component_id)
            
            reward_delta = self.max_reward_per_step

            # Update Congestion Map and calculate HPWL Penalty
            hpwl_penalty = self._update_and_score_hpwl(action.component_id)
            reward_delta -= hpwl_penalty

            if self.thermal_penalty and self._check_adjacency(action.x, action.y):
                reward_delta -= (self.max_reward_per_step * 0.5) 

        safe_reward = max(0.0, min(self.max_reward_per_step, reward_delta))
        if error_msg: safe_reward = 0.0

        is_done = len(self.unplaced_components) == 0
        info = {"error": error_msg}
        
        return self.state(), safe_reward, is_done, info

    def _update_and_score_hpwl(self, comp_id: int) -> float:
        """Calculates Half-Perimeter Wirelength and updates congestion map."""
        total_penalty = 0.0
        # Re-evaluate all nets this component is a part of
        for net in self.netlist:
            if comp_id in net:
                placed_nodes_in_net = [node for node in net if node in self.placed_locations]
                if len(placed_nodes_in_net) > 1: # Can only form a box with 2+ points
                    xs = [self.placed_locations[node][0] for node in placed_nodes_in_net]
                    ys = [self.placed_locations[node][1] for node in placed_nodes_in_net]
                    
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    
                    hpwl = (max_x - min_x) + (max_y - min_y)
                    
                    # Target ideal HPWL is roughly the number of nodes in the net
                    ideal_hpwl = len(net) 
                    if hpwl > ideal_hpwl:
                        # Penalty scales with how far the HPWL is stretched
                        total_penalty += ((hpwl - ideal_hpwl) * 0.02 * self.max_reward_per_step)

                    # Update congestion map (mark bounding box area as congested)
                    for y in range(min_y, max_y + 1):
                        for x in range(min_x, max_x + 1):
                            self.congestion_map[y][x] += 1
                            
        return total_penalty

    def _check_adjacency(self, x: int, y: int) -> bool:
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.grid[ny][nx] != 0:
                return True
        return False
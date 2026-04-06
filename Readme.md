# OpenEnv EDA Floorplanning Benchmark

## Description
This environment simulates Electronic Design Automation (EDA) macro placement. Finding optimal physical layouts for logical circuits is a real-world bottleneck in chip design. Agents must place components on a silicon grid to minimize Manhattan wirelength between connected nodes while adhering to thermal/spatial constraints.

## Action Space
Defined by `EDAAction` (Pydantic). 
- `component_id` (int): The ID of the component to place.
- `x` (int): Grid column index.
- `y` (int): Grid row index.

## Observation Space
Defined by `EDAObservation` (Pydantic).
- `grid_size` (int): Dimensions of the grid (e.g., 8 means 8x8).
- `grid_state` (List[List[int]]): 2D array where 0 is empty and integers represent placed `component_ids`.
- `unplaced_components` (List[int]): IDs remaining to be placed.
- `netlist` (List[List[int]]): Graph edges representing required wire connections.

## Tasks
1. **place_basic (Easy):** 3 components, 5x5 grid. Pure spatial reasoning. Avoid overlapping.
2. **place_routed (Medium):** 5 components, 8x8 grid. Introduces netlist routing logic. Distance between connected nodes penalizes the reward.
3. **place_thermal (Hard):** 8 components, 10x10 grid. Netlist + thermal constraints. Placing components in adjacent cells incurs massive penalties.

## Execution
Run the automated baseline:
```bash
export HF_TOKEN="your_token"
python inference.py
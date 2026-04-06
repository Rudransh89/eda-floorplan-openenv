EDA Floorplan & HPWL Routing Benchmark
======================================

### ⚡ An OpenEnv Reinforcement Learning Environment for VLSI Design

🔬 Overview
-----------

This environment benchmarks the spatial reasoning and optimization capabilities of Large Language Models (LLMs) in the context of **Electronic Design Automation (EDA)**. The agent acts as a physical design engineer, placing silicon components on a grid while minimizing routing congestion and wirelength.

🛠️ Technical Features
----------------------

This benchmark moves beyond simple "puzzles" by incorporating industry-standard EDA metrics:

-   **Half-Perimeter Wirelength (HPWL):** Uses bounding-box math to estimate the routing cost for multi-terminal nets.

-   **Thermal Adjacency Penalties:** Components placed too close to each other incur a "heat" penalty, simulating thermal management constraints.

-   **Dynamic Congestion Mapping:** Every placement updates a global congestion map, forcing the AI to balance wirelength against routing density.

-   **Chain-of-Thought (CoT) Reasoning:** The environment requires the model to explain its spatial logic before outputting coordinates, significantly reducing "hallucinated" collisions.

📊 Benchmark Tasks
------------------

The environment scales across three difficulty tiers:

1.  **`place_basic` (Easy):** 5x5 grid with simple point-to-point nets. Tests basic coordinate validity.

2.  **`place_routed` (Medium):** 8x8 grid with multi-pin nets. Focuses on minimizing total HPWL.

3.  **`place_thermal` (Hard):** 10x10 grid with high-density nets and strict thermal adjacency rules. Tests complex constraint satisfaction.

* * * * *

🚀 Installation & Local Setup
-----------------------------

### 1\. Clone the Repository

Bash

```
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

```

### 2\. Set Up the Environment

We recommend using **Conda** or **uv** for dependency management.

Bash

```
# Using uv (Recommended for OpenEnv)
pip install uv
uv lock
pip install -e .

```

### 3\. Run Validation

To ensure your environment meets the strict Meta/Hugging Face specifications:

Bash

```
openenv validate

```

* * * * *

🤖 Running the AI Agent
-----------------------

To test the baseline agent locally, you must provide your Hugging Face token:

Bash

```
export HF_TOKEN="your_huggingface_token_here"
export EDA_FLOORPLAN_TASK="place_thermal"
python inference.py

```

### 📈 Visualization

To see a graphical representation of the final layout, component placement, and the dotted HPWL bounding boxes:

Bash

```
python visualize.py

```

* * * * *

📦 Project Structure
--------------------

-   `eda_env.py`: The core Gymnasium-style physics engine and HPWL calculator.

-   `inference.py`: The AI controller with error feedback and CoT prompting.

-   `server/app.py`: FastAPI implementation for remote multi-mode deployment.

-   `openenv.yaml`: Task definitions and environment metadata for the grading bot.

-   `pyproject.toml`: Modern Python packaging and entry points.

-   `visualize.py`: Matplotlib-based GUI for human review.

📄 License
----------

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

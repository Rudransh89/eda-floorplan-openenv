import os
import sys
from fastapi import FastAPI, HTTPException
import uvicorn

# Ensure the root directory is in the path so we can import eda_env
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eda_env import EDAFloorplanEnv, EDAAction

app = FastAPI()

# Initialize environment defaulting to the bot's environment variable
task_name = os.getenv("EDA_FLOORPLAN_TASK", "place_basic")
env = EDAFloorplanEnv(task_name=task_name)

@app.get("/")
def ping():
    """Health check for Hugging Face Spaces."""
    return {"status": "ok"}

@app.post("/reset")
def reset():
    """Reset endpoint."""
    obs = env.reset()
    return obs.model_dump() if hasattr(obs, 'model_dump') else obs.dict()

@app.post("/step")
def step(action: EDAAction):
    """Step endpoint."""
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump() if hasattr(obs, 'model_dump') else obs.dict(),
            "reward": float(reward),
            "done": bool(done),
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state():
    """State endpoint."""
    obs = env.state()
    return obs.model_dump() if hasattr(obs, 'model_dump') else obs.dict()

def main():
    # Read the port assigned by Hugging Face Spaces (default 7860)
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
import os
import json
import textwrap
from openai import OpenAI
from eda_env import EDAFloorplanEnv, EDAAction

# --- OPENENV MANDATORY VARIABLES ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
TASK_NAME = os.getenv("EDA_FLOORPLAN_TASK", "place_thermal")  # Default to the most complex task for inference
BENCHMARK = "eda_floorplan"

MAX_STEPS = 15
SUCCESS_SCORE_THRESHOLD = 0.8

# --- VLM-READY CHAIN-OF-THOUGHT PROMPT ---
SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous Electronic Design Automation (EDA) layout agent.
    Goal: Place components on a silicon grid to minimize Half-Perimeter Wirelength (HPWL) and avoid high congestion areas.
    You will receive:
    1. Grid State (0 is empty, positive integers are occupied cells).
    2. Congestion Map (Higher numbers mean dense routing traffic).
    3. Netlist (Arrays of component IDs that must be wired together).
    
    CRITICAL INSTRUCTIONS:
    1. You MUST check the Grid State to ensure your chosen (x, y) coordinates contain a '0'. DO NOT place a component on a non-zero cell.
    2. Analyze the Netlist to place components close to their connected partners.
    
    Respond ONLY with a raw JSON object in this exact format: 
    {"reasoning": "Briefly explain checking the grid for an empty cell and your HPWL strategy", "component_id": int, "x": int, "y": int}
""")

def run_inference():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = EDAFloorplanEnv(task_name=TASK_NAME)
    obs = env.reset()
    
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")
    
    done = False
    step_count = 0
    total_score = 0.0
    reward_history = []
    last_error_msg = None  # Tracks the error for the feedback loop
    
    while not done and step_count < MAX_STEPS:
        step_count += 1
        error_msg = "null"
        reward = 0.0
        action_str = "none"
        
        # --- SHORT-TERM MEMORY FEEDBACK ---
        feedback = f"\nWARNING: Your last action failed with '{last_error_msg}'. Correct your mistake." if last_error_msg else ""
        
        user_prompt = (
            f"Netlist: {obs.netlist}\n"
            f"Unplaced Components: {obs.unplaced_components} (CRITICAL: You MUST select an ID from this exact list)\n"
            f"Grid State:\n{json.dumps(obs.grid_state)}\n"
            f"Congestion Map:\n{json.dumps(obs.congestion_map)}\n"
            f"{feedback}\n"
            "Action:"
        )

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, 
                max_tokens=400  # Gives the AI room to write its reasoning before outputting the coordinates
            )
            
            raw_reply = response.choices[0].message.content.strip()
            
            # Clean up potential markdown formatting from the LLM
            if raw_reply.startswith("```json"): 
                raw_reply = raw_reply[7:-3].strip()
            elif raw_reply.startswith("```"): 
                raw_reply = raw_reply[3:-3].strip()
            
            action_data = json.loads(raw_reply)
            
            # Map to Pydantic Model (ignores the 'reasoning' key automatically)
            action = EDAAction(
                component_id=action_data["component_id"], 
                x=action_data["x"], 
                y=action_data["y"]
            )
            action_str = f"place({action.component_id},{action.x},{action.y})"
            
            obs, reward, done, info = env.step(action)
            
            # --- ERROR CAPTURE ---
            if info.get("error"): 
                error_msg = f"'{info['error'].replace(' ', '_')}'"
                last_error_msg = info['error']  # Save raw error for the next prompt
            else:
                last_error_msg = None
                
        except json.JSONDecodeError:
            error_msg = "'json_parse_error'"
            last_error_msg = "You output invalid JSON. You must output a valid JSON object."
            # We don't set done=True here anymore, we let it retry!
        except Exception as e:
            error_msg = f"'{str(e).replace(' ', '_')}'"
            done = True
            
        total_score += reward
        reward_history.append(f"{reward:.2f}")
        
        print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_msg}")

    total_score = min(1.0, max(0.0, total_score))
    success = total_score >= SUCCESS_SCORE_THRESHOLD
    rewards_str = ",".join(reward_history)
    print(f"[END] success={str(success).lower()} steps={step_count} score={total_score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    run_inference()
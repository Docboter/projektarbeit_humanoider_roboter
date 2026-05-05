import os
import json
import numpy as np

def convert_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for ep in os.listdir(input_dir):
        ep_path = os.path.join(input_dir, ep)
        if not os.path.isdir(ep_path):
            continue

        states_path = os.path.join(ep_path, "states.npy")
        actions_path = os.path.join(ep_path, "actions.npy")

        if not os.path.exists(states_path):
            continue

        states = np.load(states_path)
        actions = np.load(actions_path)

        out = []
        for s, a in zip(states, actions):
            out.append({
                "state": s.tolist(),
                "action": a.tolist()
            })

        with open(os.path.join(output_dir, f"{ep}.json"), "w") as f:
            json.dump(out, f)

if __name__ == "__main__":
    convert_dataset("/data/raw", "/data/processed")
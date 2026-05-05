import os
import json
import numpy as np

def convert_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    episodes = sorted(os.listdir(input_dir))
    for ep in episodes:
        ep_path = os.path.join(input_dir, ep)
        if not os.path.isdir(ep_path):
            continue

        states = np.load(os.path.join(ep_path, "states.npy"))
        actions = np.load(os.path.join(ep_path, "actions.npy"))

        data = []
        for s, a in zip(states, actions):
            data.append({
                "state": s.tolist(),
                "action": a.tolist()
            })

        with open(os.path.join(output_dir, f"{ep}.json"), "w") as f:
            json.dump(data, f)

if __name__ == "__main__":
    convert_dataset(
        input_dir="/workspace/data/raw",
        output_dir="/workspace/data/data_lerobot"
    )
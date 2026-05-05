import os
import json
import torch
from torch.utils.data import Dataset

class RobotDataset(Dataset):
    def __init__(self, data_dir, seq_length):
        self.files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".json")
        ]
        self.seq_length = seq_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], "r") as f:
            data = json.load(f)

        states = torch.tensor([d["state"] for d in data], dtype=torch.float32)
        actions = torch.tensor([d["action"] for d in data], dtype=torch.float32)

        return states[:self.seq_length], actions[:self.seq_length]
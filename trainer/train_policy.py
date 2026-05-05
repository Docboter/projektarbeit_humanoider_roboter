import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "/workspace/data"
OUTPUT_PATH = "/workspace/output"

# -----------------------
# Dataset Wrapper
# -----------------------
class RobotDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.ds = dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]

        # ⚠️ ggf. anpassen je nach Dataset-Feldern
        state = np.array(sample["qpos"], dtype=np.float32)
        action = np.array(sample["action"], dtype=np.float32)

        return torch.tensor(state), torch.tensor(action)


# -----------------------
# Einfaches Policy Model
# -----------------------
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------
# Training
# -----------------------
def main():
    print("Loading dataset...")
    ds = load_from_disk(DATA_PATH)

    dataset = RobotDataset(ds)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    sample_state, sample_action = dataset[0]
    state_dim = sample_state.shape[0]
    action_dim = sample_action.shape[0]

    print("State dim:", state_dim)
    print("Action dim:", action_dim)

    model = PolicyNet(state_dim, action_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(10):
        total_loss = 0

        for state, action in loader:
            state = state.to(DEVICE)
            action = action.to(DEVICE)

            pred = model(state)
            loss = loss_fn(pred, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss {total_loss:.4f}")

        # Save checkpoint
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        torch.save(model.state_dict(), f"{OUTPUT_PATH}/model_epoch_{epoch}.pt")


if __name__ == "__main__":
    main()
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_utils import RobotDataset


class SimpleGroot(nn.Module):
    def __init__(self, state_dim=64, action_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train(cfg):
    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["training"]["output_dir"], exist_ok=True)

    dataset = RobotDataset(
        cfg["data"]["data_dir"],
        cfg["data"]["seq_length"]
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"]
    )

    model = SimpleGroot(
        action_dim=cfg["robot"]["action_dim"]
    ).to(cfg["model"]["device"])

    # pretrained laden (wenn vorhanden)
    if os.path.exists(cfg["model"]["pretrained_path"]):
        model.load_state_dict(
            torch.load(cfg["model"]["pretrained_path"])
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    loss_fn = nn.MSELoss()

    for epoch in range(cfg["training"]["epochs"]):
        for i, (states, actions) in enumerate(loader):
            states = states.to(cfg["model"]["device"])
            actions = actions.to(cfg["model"]["device"])

            pred = model(states)
            loss = loss_fn(pred, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % cfg["training"]["log_interval"] == 0:
                print(f"Epoch {epoch} Step {i} Loss {loss.item()}")

        # Checkpoint speichern
        torch.save(
            model.state_dict(),
            f"{cfg['training']['checkpoint_dir']}/epoch_{epoch}.pt"
        )

    # finales Modell
    torch.save(
        model.state_dict(),
        f"{cfg['training']['output_dir']}/final_model.pt"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
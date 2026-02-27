import torch
from torch.amp import GradScaler, autocast
import pandas as pd
import matplotlib.pyplot as plt


def quantile_loss(predictions, targets, patch_size=10, quantiles=[0.1, 0.5, 0.9]):
    """
    predictions: [Batch, Future_Patches, 3]  (e.g., [B, 24, 3])
    targets:     [Batch, Future_Mins]        (e.g., [B, 240])
    """
    B, _ = targets.shape

    # 1. Patch the Target
    # Reshape 240 mins into 24 patches of 10 mins, then take the mean of each patch
    # Shape becomes: [Batch, 10]
    targets_patched = targets.view(B, -1, patch_size).mean(dim=-1)

    losses = []
    for i, q in enumerate(quantiles):
        # 2. Calculate error for the entire sequence at once
        error = targets_patched - predictions[:, :, i]  # [Batch, 10]

        # 3. Asymmetric penalty
        loss = torch.max((q - 1) * error, q * error)
        losses.append(loss)

    # Stack to [Batch, 8, 3], sum over quantiles, mean over sequence and batch
    total_loss = torch.mean(torch.sum(torch.stack(losses, dim=-1), dim=-1))
    return total_loss


class TemporalFusionTrainer:
    def __init__(
        self, model, train_loader, val_loader, optimizer, scheduler, device="cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = GradScaler()
        self.device = device
        self.history = {"train_loss": [], "val_loss": []}

        # April 17, 2018 filter
        self.bad_start = pd.to_datetime("2018-04-17 08:50:00").timestamp()
        self.bad_end = pd.to_datetime("2018-04-17 09:50:00").timestamp()

    def train_epoch(self):
        self.model.train()
        total_loss, batches = 0, 0
        for batch in self.train_loader:
            x_past_power, x_past_time, x_future_time, x_stat, y = (
                batch["x_past_power"].to(self.device),
                batch["x_past_time"].to(self.device),
                batch["x_future_time"].to(self.device),
                batch["x_static"].to(self.device),
                batch["y"].to(self.device),
            )

            self.optimizer.zero_grad()
            if self.device == "cuda":
                with autocast(self.device):
                    quantiles = self.model(
                        x_past_power, x_past_time, x_future_time, x_stat
                    )
                    loss = quantile_loss(quantiles, y)
            else:
                quantiles = self.model(x_past_power, x_past_time, x_future_time, x_stat)
                loss = quantile_loss(quantiles, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # self.scheduler.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        self.history["train_loss"].append(avg_loss)
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss, batches = 0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                x_past_power, x_past_time, x_future_time, x_stat, y = (
                    batch["x_past_power"].to(self.device),
                    batch["x_past_time"].to(self.device),
                    batch["x_future_time"].to(self.device),
                    batch["x_static"].to(self.device),
                    batch["y"].to(self.device),
                )
                quantiles, _, _ = self.model(
                    x_past_power, x_past_time, x_future_time, x_stat
                )
                loss = quantile_loss(quantiles, y)
                total_loss += loss.item()
                batches += 1

        avg_loss = total_loss / max(1, batches)
        self.history["val_loss"].append(avg_loss)
        return avg_loss

    def plot_learning_curves(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.title("Temporal Fusion Model Convergence")
        plt.xlabel("Epoch")
        plt.legend()
        plt.grid(color="grey", linestyle="-", linewidth=0.5)
        plt.ylim(0, 5)
        plt.ylabel("NLL Loss")
        plt.savefig("LearningCurve.png")

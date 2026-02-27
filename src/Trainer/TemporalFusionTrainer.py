import torch
from torch.amp import GradScaler, autocast
import pandas as pd
import matplotlib.pyplot as plt


def quantile_loss(predictions, targets, quantiles=[0.1, 0.5, 0.9]):
    """
    predictions: [Batch, 3]
    targets: [Batch, 1]
    """
    losses = []
    for i, q in enumerate(quantiles):
        # Calculate error for this specific quantile
        error = targets.squeeze() - predictions[:, i]

        # Asymmetric penalty
        loss = torch.max((q - 1) * error, q * error)
        losses.append(loss)

    # Sum over quantiles, mean over batch
    return torch.mean(torch.sum(torch.stack(losses, dim=1), dim=1))


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
            x_dyn, x_stat, target = (
                batch["x_dynamic"].to(self.device),
                batch["x_static"].to(self.device),
                batch["target"].to(self.device),
            )

            self.optimizer.zero_grad()
            with autocast(self.device):
                quantiles = self.model(x_dyn, x_stat)
                loss = quantile_loss(quantiles, target)

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
                x_dyn, x_stat, target = (
                    batch["x_dynamic"].to(self.device),
                    batch["x_static"].to(self.device),
                    batch["target"].to(self.device),
                )
                # quantiles_to_compute = torch.tensor([0.1, 0.5, 0.9])
                # target_quartiles = torch.quantile(target, quantiles_to_compute)
                quantiles, _, _ = self.model(x_dyn, x_stat)
                loss = quantile_loss(quantiles, target)
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

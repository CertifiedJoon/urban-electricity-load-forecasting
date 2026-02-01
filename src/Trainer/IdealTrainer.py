import torch
from torch.cuda.amp import GradScaler, autocast
import pandas as pd


class IdealTrainer:
    def __init__(
        self, model, train_loader, val_loader, optimizer, scheduler, device="cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler()
        self.device = device

        # Datasheet specific: April 17, 2018 server error period (UTC)
        self.unreliable_start = pd.to_datetime("2018-04-17 08:50:00").timestamp()
        self.unreliable_end = pd.to_datetime("2018-04-17 09:50:00").timestamp()

    def nll_loss(self, mu, sigma, target):
        dist = torch.distributions.Normal(mu, sigma)
        return -dist.log_prob(target).mean()

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in self.train_loader:
            # --- Hard Filter Logic ---
            # Check if current batch timestamps overlap the unreliable period
            # (Assuming your dataset passes 'start_timestamp' in the batch dict)
            batch_start = batch["start_time"].min().item()
            batch_end = batch["end_time"].max().item()

            if not (
                batch_end < self.unreliable_start or batch_start > self.unreliable_end
            ):
                continue  # Skip this batch entirely if it touches the error window

            x_dyn = batch["x_dynamic"].to(self.device)
            x_stat = batch["x_static"].to(self.device)
            target = batch["target"].to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                mu, sigma = self.model(x_dyn, x_stat)
                loss = self.nll_loss(mu, sigma, target)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Step the scheduler after every batch for OneCycleLR
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

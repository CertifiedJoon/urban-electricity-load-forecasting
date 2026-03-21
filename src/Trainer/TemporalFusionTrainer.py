import torch
from torch.amp import GradScaler, autocast
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricSpikeQuantileLoss(nn.Module):
    def __init__(
        self,
        quantiles=[0.1, 0.5, 0.9],
        z_threshold=1.5,
        brave_multiplier=100.0,
        patch_size=10,
    ):
        super().__init__()
        self.quantiles = quantiles
        # z_threshold is in standard deviations (e.g., 1.5 = top ~7% of data)
        self.z_threshold = z_threshold
        self.brave_multiplier = brave_multiplier
        self.patch_size = patch_size

    def forward(self, preds, target):
        device = preds.device
        B, _ = target.shape

        # Patching the targets to match the sequence length of preds
        targets_patched = target.view(B, -1, self.patch_size).mean(dim=-1)

        # 1. Expand dimensions for broadcasting
        q_tensor = torch.tensor(self.quantiles, device=device).view(1, 1, -1)
        target_expanded = targets_patched.unsqueeze(-1)

        # 2. Calculate raw errors
        errors = target_expanded - preds

        # 3. Standard Pinball Loss
        standard_loss = torch.max(q_tensor * errors, (q_tensor - 1) * errors)

        # --- THE FIX: BI-DIRECTIONAL ASYMMETRIC WEIGHTING ---

        # Condition A: Is this an extreme event? (Operating in Z-score space)
        is_peak = (target_expanded > self.z_threshold).float()
        is_drop = (target_expanded < -self.z_threshold).float()
        is_extreme = is_peak + is_drop  # Will be 1.0 if either is true

        # Condition B: Was the model cowardly?
        # For peaks, cowardly means under-predicting (Target > Pred -> Error > 0)
        cowardly_peak = is_peak * (errors > 0).float()

        # For drops, cowardly means over-predicting (Target < Pred -> Error < 0)
        cowardly_drop = is_drop * (errors < 0).float()

        is_cowardly = cowardly_peak + cowardly_drop

        # Condition C: What is the magnitude of the extreme event?
        # How far past the threshold did it actually go?
        peak_magnitude = F.relu(target_expanded - self.z_threshold)
        drop_magnitude = F.relu(-self.z_threshold - target_expanded)
        extreme_magnitude = peak_magnitude + drop_magnitude

        # Build the Multiplier
        # If the model was cowardly during an extreme event, punish it heavily.
        # If the model was brave (overshot a peak or undershot a drop), multiplier remains a safe 1.0.
        multiplier = 1.0 + (is_cowardly * self.brave_multiplier * extreme_magnitude)

        # 4. Apply weights and average across the 3 quantiles -> Shape: [B, seq_len]
        weighted_loss = standard_loss * multiplier
        weighted_loss = torch.mean(weighted_loss, dim=2)

        # --- THE TEMPORAL DILUTION FIX (SPLIT-MEAN) ---

        # Squeeze the trailing dimension off our extreme mask so it matches weighted_loss
        is_extreme_2d = is_extreme.squeeze(-1)

        # Flatten the tensors to separate the timesteps cleanly
        flat_loss = weighted_loss.view(-1)
        flat_is_extreme = is_extreme_2d.view(-1)

        # 1. Calculate the mean of ONLY the normal, boring timesteps
        normal_timesteps_loss = flat_loss[flat_is_extreme == 0.0]
        mean_normal_loss = (
            torch.mean(normal_timesteps_loss) if len(normal_timesteps_loss) > 0 else 0.0
        )

        # 2. Calculate the mean of ONLY the extreme timesteps (both peaks and drops)
        extreme_timesteps_loss = flat_loss[flat_is_extreme == 1.0]
        if len(extreme_timesteps_loss) > 0:
            mean_extreme_loss = torch.mean(extreme_timesteps_loss)
        else:
            mean_extreme_loss = 0.0

        # 3. Combine them. Extreme volatility now has equal voting power to the baseline!
        final_loss = mean_normal_loss + mean_extreme_loss

        return final_loss


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
        self.loss = AsymmetricSpikeQuantileLoss()

        # April 17, 2018 filter
        self.bad_start = pd.to_datetime("2018-04-17 08:50:00").timestamp()
        self.bad_end = pd.to_datetime("2018-04-17 09:50:00").timestamp()

    def train_epoch(self, accum_step):
        self.model.train()
        total_loss, batches = 0, 0
        self.optimizer.zero_grad()
        for i, batch in enumerate(self.train_loader):
            (
                x_past_power,
                x_past_time,
                x_past_temperature,
                x_past_weather_conditions,
                x_future_time,
                x_future_temperature,
                x_future_weather_conditions,
                x_stat,
                y,
            ) = (
                batch["x_past_power"].to(self.device),
                batch["x_past_time"].to(self.device),
                batch["x_past_temperature"].to(self.device),
                batch["x_past_weather_conditions"].to(self.device),
                batch["x_future_time"].to(self.device),
                batch["x_future_temperature"].to(self.device),
                batch["x_future_weather_conditions"].to(self.device),
                batch["x_static"].to(self.device),
                batch["y"].to(self.device),
            )

            if self.device == "cuda":
                with autocast(self.device):
                    quantiles = self.model(
                        x_past_power,
                        x_past_time,
                        x_past_temperature,
                        x_past_weather_conditions,
                        x_future_time,
                        x_future_temperature,
                        x_future_weather_conditions,
                        x_stat,
                    )
                    loss = self.loss(quantiles, y) / accum_step
            else:
                quantiles = self.model(
                    x_past_power,
                    x_past_time,
                    x_past_temperature,
                    x_past_weather_conditions,
                    x_future_time,
                    x_future_temperature,
                    x_future_weather_conditions,
                    x_stat,
                )
                loss = self.loss(quantiles, y)

            self.scaler.scale(loss).backward()

            if (i + 1) % accum_step == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        self.history["train_loss"].append(avg_loss)
        return avg_loss

    def validate(self, accum_step):
        self.model.eval()
        total_loss, batches = 0, 0
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                (
                    x_past_power,
                    x_past_time,
                    x_past_temperature,
                    x_past_weather_conditions,
                    x_future_time,
                    x_future_temperature,
                    x_future_weather_conditions,
                    x_stat,
                    y,
                ) = (
                    batch["x_past_power"].to(self.device),
                    batch["x_past_time"].to(self.device),
                    batch["x_past_temperature"].to(self.device),
                    batch["x_past_weather_conditions"].to(self.device),
                    batch["x_future_time"].to(self.device),
                    batch["x_future_temperature"].to(self.device),
                    batch["x_future_weather_conditions"].to(self.device),
                    batch["x_static"].to(self.device),
                    batch["y"].to(self.device),
                )
                quantiles, _, _ = self.model(
                    x_past_power,
                    x_past_time,
                    x_past_temperature,
                    x_past_weather_conditions,
                    x_future_time,
                    x_future_temperature,
                    x_future_weather_conditions,
                    x_stat,
                )
                loss = self.loss(quantiles, y) / accum_step
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
        plt.ylim(0, 0.1)
        plt.ylabel("NLL Loss")
        plt.savefig("LearningCurve.png")

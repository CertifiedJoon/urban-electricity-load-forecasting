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
        brave_multiplier=None,
        w_peak=2.0,
        w_trough=2.0,
        patch_size=10,
        baseline=False,
    ):
        super().__init__()
        self.baseline = baseline
        self.quantiles = quantiles
        # z_threshold is in standard deviations (e.g., 1.5 = top ~7% of data)
        self.z_threshold = z_threshold
        if brave_multiplier is not None:
            w_peak = brave_multiplier
            w_trough = brave_multiplier
        self.w_peak = w_peak
        self.w_trough = w_trough
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
        if self.baseline:
            return torch.mean(standard_loss.view(-1))

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

        # Condition C: What is the magnitude of the extreme event?
        # How far past the threshold did it actually go?
        peak_magnitude = F.relu(target_expanded - self.z_threshold)
        drop_magnitude = F.relu(-self.z_threshold - target_expanded)
        extreme_magnitude = peak_magnitude + drop_magnitude

        # Build the Multiplier
        # If the model was cowardly during an extreme event, punish it heavily.
        # If the model was brave (overshot a peak or undershot a drop), multiplier remains a safe 1.0.
        peak_multiplier = 1.0 + (cowardly_peak * self.w_peak * peak_magnitude)
        trough_multiplier = 1.0 + (cowardly_drop * self.w_trough * drop_magnitude)
        multiplier = torch.maximum(peak_multiplier, trough_multiplier)

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
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        result_path,
        device="cuda",
        baseline=False,
        w_peak=2.0,
        w_trough=2.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = GradScaler(enabled=str(device).startswith("cuda"))
        self.device = device
        self.history = {"train_loss": [], "val_loss": []}
        self.loss = AsymmetricSpikeQuantileLoss(
            baseline=baseline,
            w_peak=w_peak,
            w_trough=w_trough,
        )
        self.result_path = result_path

        # April 17, 2018 filter
        self.bad_start = pd.to_datetime("2018-04-17 08:50:00").timestamp()
        self.bad_end = pd.to_datetime("2018-04-17 09:50:00").timestamp()

    def _forward_quantiles(self, *model_inputs):
        outputs = self.model(*model_inputs)
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

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

            if str(self.device).startswith("cuda"):
                with autocast(device_type="cuda"):
                    quantiles = self._forward_quantiles(
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
            else:
                quantiles = self._forward_quantiles(
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

            loss_for_backward = loss / accum_step
            self.scaler.scale(loss_for_backward).backward()

            if (i + 1) % accum_step == 0 or (i + 1) == len(self.train_loader):
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
                if str(self.device).startswith("cuda"):
                    with autocast(device_type="cuda"):
                        quantiles = self._forward_quantiles(
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
                else:
                    quantiles = self._forward_quantiles(
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
                total_loss += loss.item()
                batches += 1

        avg_loss = total_loss / max(1, batches)
        self.history["val_loss"].append(avg_loss)
        return avg_loss

    def plot_learning_curves(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        epochs = range(1, len(self.history["train_loss"]) + 1)
        ax.plot(epochs, self.history["train_loss"], label="Train Loss")
        ax.plot(epochs, self.history["val_loss"], label="Val Loss")
        ax.set_title("Temporal Fusion Model Convergence")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

        all_losses = self.history["train_loss"] + self.history["val_loss"]
        if all_losses:
            min_loss = min(all_losses)
            max_loss = max(all_losses)
            spread = max_loss - min_loss
            if spread == 0:
                pad = max(abs(max_loss) * 0.05, 1e-4)
            else:
                pad = max(spread * 0.12, 1e-4)
            ax.set_ylim(min_loss - pad, max_loss + pad)

        fig.tight_layout()
        fig.savefig(self.result_path + "/LearningCurve.png")
        plt.close(fig)

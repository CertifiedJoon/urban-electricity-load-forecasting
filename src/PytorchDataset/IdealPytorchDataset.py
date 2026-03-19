from collections import deque

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class IdealPytorchDataset(Dataset):
    def __init__(
        self,
        home_ids,
        orchestrator,
        split="train",
        window_size=43200,
        prediction_shift=240,
        train_pct=0.80,
        val_pct=0.10,
        power_stats=None,
        weather_stats=None,
        loss_momentum_length=10,
    ):
        self.split = split
        self.window_size = window_size
        self.prediction_shift = prediction_shift
        self.samples = []

        # Deadzone equals history + lead time to prevent future leakage
        deadzone = pd.Timedelta(minutes=(window_size + prediction_shift))

        print(f"Loading '{split}' split for {len(home_ids)} homes...")

        for h_id in home_ids:
            # Orchestrator now simply joins and returns the FULL timeline for the home
            static_data, full_dynamic_df = orchestrator.get_home_data(h_id)

            if full_dynamic_df is None or full_dynamic_df.empty:
                continue

            start_time = full_dynamic_df.index.min()
            end_time = full_dynamic_df.index.max()
            total_duration = end_time - start_time

            # Skip if the home didn't survive long enough to have distinct splits separated by deadzones
            if total_duration < (deadzone * 3):
                continue

            # --- 1. CALCULATE ROLLING CUTOFFS PER HOME ---
            train_end_time = start_time + (total_duration * train_pct)
            val_start_time = train_end_time + deadzone
            val_end_time = start_time + (total_duration * (train_pct + val_pct))
            test_start_time = val_end_time + deadzone

            # --- 2. SLICE DATASET ---
            if split == "train":
                split_df = full_dynamic_df[
                    full_dynamic_df.index < train_end_time
                ].copy()
            elif split == "val":
                split_df = full_dynamic_df[
                    (full_dynamic_df.index >= val_start_time)
                    & (full_dynamic_df.index < val_end_time)
                ].copy()
            elif split == "test":
                split_df = full_dynamic_df[
                    full_dynamic_df.index >= test_start_time
                ].copy()
            else:
                raise ValueError(f"Invalid split: {split}")

            # Skip if the sliced chunk is too small to pull even one full window from
            if len(split_df) <= (self.window_size + self.prediction_shift):
                continue

            # Reset index for positional integer slicing in __getitem__
            split_df = split_df.reset_index(drop=True)

            self.samples.append(
                {"static": static_data, "dynamic": split_df, "homeid": h_id}
            )

        print(f"Loaded {len(self.samples)} valid homes for '{split}'.")

        # --- 3. ISOLATE STATS TO PREVENT LEAKAGE ---
        if split == "train":
            # Calculate global mean/std ONLY on the training distribution
            all_power = pd.concat([s["dynamic"]["value"] for s in self.samples])
            all_temps = pd.concat([s["dynamic"]["temperature"] for s in self.samples])

            self.power_stats = {"mean": all_power.mean(), "std": all_power.std()}
            self.weather_stats = {"mean": all_temps.mean(), "std": all_temps.std()}
        else:
            # Val and Test MUST use the Train stats passed into the constructor
            if power_stats is None or weather_stats is None:
                raise ValueError(
                    f"You must provide 'train' stats when building the '{split}' dataset!"
                )
            self.power_stats = power_stats
            self.weather_stats = weather_stats

        # track train loss
        self.loss_trend = deque()
        self.loss_momentum_length = loss_momentum_length

    def __len__(self):
        return len(self.samples)

    def denormalize(self, z_score):
        """Helper to convert standardized predictions back to kW for evaluation"""
        return (z_score * self.power_stats["std"]) + self.power_stats["mean"]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        full_dyn = sample["dynamic"]
        static_data = sample["static"]

        max_start = len(full_dyn) - self.window_size - self.prediction_shift

        if max_start <= 0:
            start_idx = 0
        else:
            # --- 4. SPLIT-AWARE SAMPLING LOGIC ---
            if self.split == "train":
                spike_threshold = full_dyn["value"].quantile(0.7)
                high_power_idx = np.where(full_dyn["value"].values > spike_threshold)[0]

                valid_spike_starts = (
                    high_power_idx
                    - self.window_size
                    - int(self.prediction_shift * np.random.rand())
                )

                spike_start_pool = valid_spike_starts[
                    (valid_spike_starts >= 0) & (valid_spike_starts <= max_start)
                ]
                max_loss_fall_pct = self._get_max_fall_pct()

                # 50/50 Chance to force a spike into the training batch
                if (
                    np.random.rand() < (1 - max_loss_fall_pct)
                    and len(spike_start_pool) > 0
                ):
                    start_idx = np.random.choice(spike_start_pool)
                else:
                    start_idx = np.random.randint(0, max_start)
            else:
                # Validation and Test MUST use random sampling for honest evaluation
                start_idx = np.random.randint(0, max_start)

        # --- 5. EXTRACT & STANDARDIZE SEQUENCES ---
        input_seq = full_dyn.iloc[start_idx : start_idx + self.window_size].copy()

        input_seq.loc[:, "value"] = (
            input_seq["value"] - self.power_stats["mean"]
        ) / self.power_stats["std"]
        input_seq.loc[:, "temperature"] = (
            input_seq["temperature"] - self.weather_stats["mean"]
        ) / self.weather_stats["std"]

        x_past_power = torch.from_numpy(input_seq["value"].values).float().unsqueeze(-1)
        x_past_temperature = (
            torch.from_numpy(input_seq["temperature"].values).float().unsqueeze(-1)
        )
        x_past_weather_conditions = torch.from_numpy(
            input_seq["conditions"].values
        ).long()
        x_past_time = torch.from_numpy(
            input_seq[["hour", "dayofweek", "month"]].values
        ).long()

        future_seq = full_dyn.iloc[
            start_idx
            + self.window_size : start_idx
            + self.window_size
            + self.prediction_shift
        ].copy()

        future_seq.loc[:, "value"] = (
            future_seq["value"] - self.power_stats["mean"]
        ) / self.power_stats["std"]
        future_seq.loc[:, "temperature"] = (
            future_seq["temperature"] - self.weather_stats["mean"]
        ) / self.weather_stats["std"]

        x_future_time = torch.from_numpy(
            future_seq[["hour", "dayofweek", "month"]].values
        ).long()
        x_future_temperature = (
            torch.from_numpy(future_seq["temperature"].values).float().unsqueeze(-1)
        )
        x_future_weather_conditions = torch.from_numpy(
            future_seq["conditions"].values
        ).long()

        y_seq = torch.from_numpy(future_seq["value"].values).float()

        static_tensor = torch.tensor(
            [
                static_data["homeid"],
                static_data["residents"],
                static_data["income_band"],
                static_data["hometype"],
                static_data["urban_rural_class"],
                static_data["build_era"],
                static_data["occupied_days"],
                static_data["occupied_nights"],
                static_data["workingstatus"],
                static_data["gender"],
                static_data["ageband"],
                static_data["weeklyhoursofwork"],
            ],
            dtype=torch.long,
        )

        return {
            "x_past_power": x_past_power,
            "x_past_time": x_past_time,
            "x_past_temperature": x_past_temperature,
            "x_past_weather_conditions": x_past_weather_conditions,
            "x_static": static_tensor,
            "x_future_time": x_future_time,
            "x_future_temperature": x_future_temperature,
            "x_future_weather_conditions": x_future_weather_conditions,
            "y": y_seq,
        }

    def get_full_home_stream(self, home_id):
        """
        RELEVANT FOR VISUALIZATION:
        Finds a specific home by its ID and returns the raw continuous data.
        """
        # Find the specific home in our sample list
        for s in self.samples:
            if str(s["homeid"]) == str(home_id):
                target_sample = s
                break
        target_sample = None
        if target_sample is None:
            raise ValueError(f"Home ID {home_id} not found in dataset.")

        # Standardize
        power = target_sample["dynamic"]
        power["value"] = (power["value"] - self.power_stats["mean"]) / self.power_stats[
            "std"
        ]
        power["temperature"] = (
            power["temperature"] - self.weather_stats["mean"]
        ) / self.weather_stats["std"]

        # Convert the full numpy array to a tensor of shape [Total_Mins, 1]
        full_power_tensor = torch.tensor(power["value"].values).float().unsqueeze(-1)
        full_time_tensor = torch.tensor(
            power[["hour", "dayofweek", "month"]].values
        ).long()
        full_temperature_tensor = (
            torch.tensor(power["temperature"].values).float().unsqueeze(-1)
        )
        full_weather_condition_tensor = torch.tensor(power["conditions"].values).long()

        static_data = target_sample["static"]
        # Get the static socio-economic features
        static_features = torch.tensor(
            [
                static_data["homeid"],
                static_data["residents"],
                static_data["income_band"],
                static_data["hometype"],
                static_data["urban_rural_class"],
                static_data["build_era"],
                static_data["occupied_days"],
                static_data["occupied_nights"],
                static_data["workingstatus"],
                static_data["gender"],
                static_data["ageband"],
                static_data["weeklyhoursofwork"],
            ],
            dtype=torch.long,
        )

        return (
            full_power_tensor,
            full_time_tensor,
            full_temperature_tensor,
            full_weather_condition_tensor,
            static_features,
        )

    def denormalize(self, val):
        """Converts model output back to log-scale for plotting"""
        if self.power_stats:
            return (val * self.power_stats["std"]) + self.power_stats["mean"]
        return val

    def update_loss_trend(self, loss):
        if len(self.loss_trend) >= self.loss_momentum_length:
            self.loss_trend.popleft()

        self.loss_trend.append(loss)

    def _get_max_fall_pct(self):
        max_loss = 0
        max_fall_pct = 0
        for i, loss in enumerate(self.loss_trend):
            if loss > max_loss:
                max_loss = loss
            elif max_fall_pct < (max_loss - loss) / max_loss:
                max_fall_pct = max(0, (max_loss - loss) / max_loss)
        return max_fall_pct

import torch
from torch.utils.data import Dataset
import numpy as np


class IdealPytorchDataset(Dataset):
    def __init__(self, home_ids, orchestrator, window_size=43200, prediction_shift=240):
        # We need window_size + 1 to create the shift
        self.window_size = window_size
        self.fetch_size = window_size + prediction_shift
        self.samples = []
        self.prediction_shift = prediction_shift

        print(f"Scanning directory for {len(home_ids)} homes...")
        for h_id in home_ids:
            static, dynamic = orchestrator.get_home_data(h_id)

            # Debugging: Show why a home might be skipped
            if dynamic is None:
                print(f"Home {h_id}: Skipped (No Data Found)")  # Uncomment if too noisy
                continue
            if len(dynamic) <= self.fetch_size:
                print(
                    f"Home {h_id}: Skipped (Data too short: {len(dynamic)} vs {self.fetch_size})"
                )
                continue
            # Ensure we have enough data for Input + 1 Target
            if dynamic is not None and len(dynamic) > self.fetch_size:
                self.samples.append(
                    {"static": static, "dynamic": dynamic, "homeid": h_id}
                )
        print(f"Loaded {len(self.samples)} valid homes.")

        # get global stats to use in standardization
        self.stats = calculate_global_stats(
            [sample["dynamic"] for sample in self.samples]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        full_dyn = sample["dynamic"]
        static_data = sample["static"]
        prediction_shift = self.prediction_shift

        # Random slice
        max_start = len(full_dyn) - self.fetch_size
        if max_start <= 0:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, max_start)

        # Grab window + shift extra step
        window = full_dyn.iloc[start_idx : start_idx + self.fetch_size]
        # standardize
        window = (window - self.stats["mean"]) / self.stats["std"]
        values = window["value"].values

        # shift 4 hours
        input_seq = values[: self.window_size]
        target = values[-1]

        static_tensor = torch.tensor(
            [
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
            "x_dynamic": torch.tensor(input_seq, dtype=torch.float32).unsqueeze(-1),
            "x_static": static_tensor,
            "target": torch.tensor([target]).float(),
            "start_ts": window.index[0].timestamp(),
            "end_ts": window.index[
                -prediction_shift - 1
            ].timestamp(),  # End of input window
        }

    def get_full_home_stream(self, home_id):
        """
        RELEVANT FOR VISUALIZATION:
        Finds a specific home by its ID and returns the raw continuous data.
        """
        # Find the specific home in our sample list
        target_inx = None
        for s in self.samples:
            if str(s["homeid"]) == str(home_id):
                target_sample = s
                break

        if target_sample is None:
            raise ValueError(f"Home ID {home_id} not found in dataset.")

        # Standardize
        standardized_sample = (
            target_sample["dynamic"] - self.stats["mean"]
        ) / self.stats["std"]

        # Convert the full numpy array to a tensor of shape [Total_Mins, 1]
        full_power_tensor = torch.tensor(
            standardized_sample["value"].values, dtype=torch.float32
        ).unsqueeze(-1)

        static_data = target_sample["static"]
        # Get the static socio-economic features
        static_features = torch.tensor(
            [
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

        return full_power_tensor, static_features

    def denormalize(self, val):
        """Converts model output back to log-scale for plotting"""
        if self.stats:
            return (val * self.stats["std"]) + self.stats["mean"]
        return val


def calculate_global_stats(samples_list):
    # Combine all power data into one array to find the true population mean/std
    all_power = np.concatenate(samples_list)
    return {"mean": np.mean(all_power), "std": np.std(all_power)}

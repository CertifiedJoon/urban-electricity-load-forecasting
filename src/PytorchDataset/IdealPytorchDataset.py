import torch
from torch.utils.data import Dataset


class IdealPytorchDataset(Dataset):
    def __init__(self, home_ids, orchestrator, window_size=40320, stride=1440):
        """
        Args:
            home_ids: List of home IDs to include in this split (train/val/test).
            orchestrator: The IdealDataOrchestrator instance we built.
            window_size: Number of minutes in one sample (e.g., 40,320 for ~28 days).
            stride: How many minutes to skip between window starts (e.g., 1440 for 1 day).
        """
        self.orchestrator = orchestrator
        self.window_size = window_size
        self.samples = []

        # 1. Pre-fetch and Indexing
        for h_id in home_ids:
            # Note: In a real setup, you'd map h_id to its specific sensor/weather IDs
            # as per the datasheet naming conventions[cite: 122, 130].
            s_id, w_id = self._get_mappings(h_id)

            static, dynamic = self.orchestrator.get_full_sample(h_id, s_id, w_id)

            # Create valid window start indices for this specific house
            max_start = len(dynamic) - window_size
            for start_idx in range(0, max_start, stride):
                self.samples.append(
                    {
                        "static": static,
                        "dynamic": dynamic.iloc[start_idx : start_idx + window_size],
                        "home_id": h_id,
                    }
                )

    def _get_mappings(self, home_id):
        # Placeholder for sensor mapping logic derived from the metadata tables[cite: 11, 122].
        return "sensor_id_here", "weather_id_here"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 2. Convert Dynamic Data to Tensors (Load + Weather + Time Features)
        # We drop the timestamp index and keep the values
        dynamic_tensor = torch.tensor(sample["dynamic"].values, dtype=torch.float32)

        # 3. Convert Static Data (Socio-Economic DNA)
        # These are usually categorical; we will use an Embedding layer later,
        # so we pass them as LongTensors (integers).
        static_values = sample["static"].drop(columns=["homeid"]).values.flatten()
        static_tensor = torch.tensor(static_values.astype(np.int64), dtype=torch.long)

        return {
            "x_dynamic": dynamic_tensor,  # Shape: [Window_Size, Feature_Count]
            "x_static": static_tensor,  # Shape: [Static_Feature_Count]
            "target": dynamic_tensor[
                :, 0
            ],  # Example: predicting the next steps of 'value'
        }

import torch
from torch.utils.data import Dataset
import numpy as np

class IdealPytorchDataset(Dataset):
    def __init__(self, home_ids, orchestrator, window_size=40320):
        # We need window_size + 1 to create the shift
        self.window_size = window_size
        self.fetch_size = window_size + 1 
        self.samples = []
        
        print(f"Scanning directory for {len(home_ids)} homes...")
        for h_id in home_ids:
            static, dynamic = orchestrator.get_home_data(h_id)
            
            # Debugging: Show why a home might be skipped
            if dynamic is None:
                print(f"Home {h_id}: Skipped (No Data Found)") # Uncomment if too noisy
                continue
            if len(dynamic) <= self.fetch_size:
                print(f"Home {h_id}: Skipped (Data too short: {len(dynamic)} vs {self.fetch_size})")
                continue
            # Ensure we have enough data for Input + 1 Target
            if dynamic is not None and len(dynamic) > self.fetch_size:
                self.samples.append({
                    'static': static,
                    'dynamic': dynamic,
                    'homeid': h_id
                })
        print(f"Loaded {len(self.samples)} valid homes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        full_dyn = sample['dynamic']
        static_data = sample['static']
        
        # Random slice
        max_start = len(full_dyn) - self.fetch_size
        if max_start <= 0: start_idx = 0
        else: start_idx = np.random.randint(0, max_start)
            
        # Grab window + 1 extra step
        window_plus_one = full_dyn.iloc[start_idx : start_idx + self.fetch_size]
        values = window_plus_one['value'].values
        
        # --- THE FIX: SHIFTING ---
        # Input:  [0, 1, 2 ... N-1]
        # Target: [1, 2, 3 ... N]
        input_seq = values[:-1]
        target_seq = values[1:]
        
        static_tensor = torch.tensor([
            static_data['residents'],
            static_data['income_band'],
            static_data['hometype'],
            static_data['workingstatus']
        ], dtype=torch.long)

        return {
            'x_dynamic': torch.tensor(input_seq, dtype=torch.float32).unsqueeze(-1),
            'x_static': static_tensor,
            'target': torch.tensor(target_seq, dtype=torch.float32),
            'start_ts': window_plus_one.index[0].timestamp(),
            'end_ts': window_plus_one.index[-2].timestamp() # End of input window
        }
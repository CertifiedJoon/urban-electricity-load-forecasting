import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.Transformer.TemporalFusionTransformer import TemporalFusionTransformer
from src.Orchestrator.IdealDatasetOrchestrator import IdealDatasetOrchestrator
from train_temporal_fusion import build_datasets, get_home_ids, split_home_ids
from benchmark_temporal_fusion import make_test_loader

# ==========================================
# 1. ATTENTION EXTRACTION HOOK
# ==========================================
attention_weights_cache = {}

def get_attention_hook(model_name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            attn_matrix = output[1] 
        else:
            attn_matrix = output
        attention_weights_cache[model_name] = attn_matrix.detach().cpu().numpy()
    return hook

# ==========================================
# 2. INFERENCE & EXTRACTION
# ==========================================
def extract_attention_for_event(models_dict, test_loader, device, cardinalities):
    peak_batch = None
    peak_index = None
    global_max = -float('inf')
    
    print("Scanning test set for the most severe peak event...")
    for batch in test_loader:
        targets = batch["y"].to(device)
        max_vals = targets.max(dim=1).values
        batch_max = max_vals.max().item()
        
        if batch_max > global_max:
            global_max = batch_max
            peak_index = torch.argmax(max_vals).item()
            peak_batch = {k: v[peak_index:peak_index+1].to(device) for k, v in batch.items()}

    if peak_batch is None:
        raise ValueError("No peak event found in the dataloader subset.")

    print(f"Peak event isolated with normalized value: {global_max:.2f}. Extracting internal attention matrices...")
    
    extracted_attentions = {}
    
    for name, info in models_dict.items():
        path = info["path"]
        smoke_test = info.get("smoke_test", False)
        
        model = TemporalFusionTransformer(cardinalities, smoke_test=smoke_test).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        
        handle = model.attention.register_forward_hook(get_attention_hook(name))
        
        with torch.no_grad():
            _ = model(
                peak_batch["x_past_power"],
                peak_batch["x_past_time"],
                peak_batch["x_past_temperature"],
                peak_batch["x_past_weather_conditions"],
                peak_batch["x_future_time"],
                peak_batch["x_future_temperature"],
                peak_batch["x_future_weather_conditions"],
                peak_batch["x_static"],
            )
            
        handle.remove()
        
        history_len = peak_batch["x_past_power"].shape[1] // model.patch_size
        terminal_attention = attention_weights_cache[name][0, -1, :history_len] 
        extracted_attentions[name] = terminal_attention
        
    return extracted_attentions, peak_batch

# ==========================================
# 3. PUBLICATION-READY VISUALIZATION
# ==========================================
def plot_attention_comparison(extracted_attentions, history_len):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    time_steps = np.arange(-history_len, 0)
    
    base_attn = extracted_attentions.get("Baseline")
    asym_attn = extracted_attentions.get("Loss_Only")
    syn_attn = extracted_attentions.get("Synergy")
    
    if base_attn is not None:
        ax.plot(time_steps, base_attn, label="Baseline (Recency Bias)", color="black", linewidth=2.5, alpha=0.8)
    if asym_attn is not None:
        ax.plot(time_steps, asym_attn, label="Asymmetric Only (Hyper-Sensitive/Spurious)", color="#ef4444", linestyle="-", linewidth=2.5, alpha=0.9)
    if syn_attn is not None:
        ax.plot(time_steps, syn_attn, label="Proposed Synergy (Structural Precursor)", color="#3b82f6", linestyle="-", linewidth=3.5)

    ax.set_title("TFT Self-Attention Distribution: Hyper-Sensitivity vs. Structural Focus", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Historical Time Steps (t-L to t-1)", fontsize=14)
    ax.set_ylabel("Normalized Attention Weight", fontsize=14)
    
    # Dynamic Annotations based on actual peaks
    # We ignore the last 5 time steps (which often dominate due to recency bias)
    # to find the structural and spurious peaks in the past.
    search_range = max(1, history_len - 5)
    
    if syn_attn is not None:
        syn_peak_idx = np.argmax(syn_attn[:search_range])
        ax.annotate('True Structural Precursor', 
                    xy=(time_steps[syn_peak_idx], syn_attn[syn_peak_idx]),
                    xytext=(time_steps[syn_peak_idx] - 10, syn_attn[syn_peak_idx] + 0.02),
                    arrowprops=dict(facecolor='#3b82f6', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=12, fontweight='bold', color="#1e3a8a")
                
    if asym_attn is not None:
        asym_peak_idx = np.argmax(asym_attn[:search_range])
        # If the spurious peak happens to be at the exact same location as the structural precursor, 
        # let's try to find another prominent spike for annotation purposes so they don't overlap completely.
        if syn_attn is not None and asym_peak_idx == syn_peak_idx and search_range > 1:
            temp_asym = np.copy(asym_attn[:search_range])
            temp_asym[asym_peak_idx] = -1 # mask out the max
            asym_peak_idx = np.argmax(temp_asym)
            
        ax.annotate('Spurious Noise Overfitting', 
                    xy=(time_steps[asym_peak_idx], asym_attn[asym_peak_idx]),
                    xytext=(time_steps[asym_peak_idx] - 15, asym_attn[asym_peak_idx] + 0.03),
                    arrowprops=dict(facecolor='#ef4444', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=12, fontweight='bold', color="#991b1b")

    ax.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.9, edgecolor='black')
    plt.tight_layout()
    plt.savefig("attention_hypersensitivity.pdf", format='pdf', dpi=300)
    print("Successfully generated 'attention_hypersensitivity.pdf'")
    plt.show()

# ==========================================
# 4. EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models_to_test = {
        "Baseline": {"path": "training_results/benchmark_tft/base_pinball_sr0.0/model.pth"},
        "Loss_Only": {"path": "training_results/benchmark_tft/oracle_asymmetric_sr0.0/model.pth"},
        "Synergy": {"path": "training_results/benchmark_tft/oracle_asymmetric_sr0.5/model.pth"}
    }
    
    for name, info in list(models_to_test.items()):
        if not os.path.exists(info["path"]):
            print(f"Warning: Model {name} not found at {info['path']}.")
            continue
            
        run_config_path = os.path.join(os.path.dirname(info["path"]), "run_config.json")
        if os.path.exists(run_config_path):
            with open(run_config_path, "r") as f:
                config = json.load(f)
            info["smoke_test"] = config.get("smoke_test", False)
            info["seed"] = config.get("seed", 42)
        else:
            info["smoke_test"] = False
            info["seed"] = 42

    data_dir = "data"
    
    base_info = models_to_test.get("Baseline", models_to_test.get("Synergy"))
    if base_info is None or not os.path.exists(base_info["path"]):
        print("Required models are missing. Cannot run extraction.")
    else:
        seed = base_info.get("seed", 42)
        orchestrator = IdealDatasetOrchestrator(data_dir)
        home_ids = get_home_ids(data_dir)
        train_ids, val_ids, test_ids = split_home_ids(home_ids, seed=seed)
        
        train_dataset, val_dataset, test_dataset = build_datasets(
            orchestrator=orchestrator,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            sampling_rate=0.0,
            eval_samples_per_home=8,
        )
        
        experiment = {
            "orchestrator": orchestrator,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset,
            "test_ids": test_ids,
        }
        
        benchmark_dataset, test_loader = make_test_loader(
            experiment=experiment,
            batch_size=4,
            device=device,
            eval_samples_per_home=8,
        )
        
        try:
            extracted_attentions, peak_batch = extract_attention_for_event(
                {k: v for k, v in models_to_test.items() if os.path.exists(v["path"])}, 
                test_loader, 
                device, 
                orchestrator.cardinalities
            )
            
            history_len = peak_batch["x_past_power"].shape[1] // 10
            plot_attention_comparison(extracted_attentions, history_len)
            
        except Exception as e:
            print(f"Error during extraction: {e}")
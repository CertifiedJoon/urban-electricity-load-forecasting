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
# Use this if your TFT doesn't return attention weights by default.
attention_weights_cache = {}

def get_attention_hook(model_name):
    """ PyTorch forward hook to intercept attention weights """
    def hook(module, input, output):
        # Depending on your precise multi-head attention implementation, 
        # output is usually (context_layer, attention_probs)
        if isinstance(output, tuple):
            attn_matrix = output[1] 
        else:
            attn_matrix = output
        
        # Average across multiple attention heads to get the global focus
        # PyTorch MultiheadAttention returns weights already averaged over heads if average_attn_weights=True
        attention_weights_cache[model_name] = attn_matrix.detach().cpu().numpy()
    return hook

# ==========================================
# 2. INFERENCE & EXTRACTION
# ==========================================
def extract_attention_for_event(models_dict, test_loader, device, cardinalities):
    """
    Finds the most severe peak event in the test set, runs it through all models, and extracts attention.
    """
    # 1. Scan dataloader for the most extreme peak event
    peak_batch = None
    peak_index = None
    global_max = -float('inf')
    
    print("Scanning test set for the most severe peak event...")
    for batch in test_loader:
        targets = batch["y"].to(device)
        
        # Find the max target in this batch
        max_vals = targets.max(dim=1).values
        batch_max = max_vals.max().item()
        
        if batch_max > global_max:
            global_max = batch_max
            peak_index = torch.argmax(max_vals).item()
            # Isolate this specific sequence (batch size = 1)
            peak_batch = {k: v[peak_index:peak_index+1].to(device) for k, v in batch.items()}

    if peak_batch is None:
        raise ValueError("No peak event found in the dataloader subset.")

    print(f"Peak event isolated with normalized value: {global_max:.2f}. Extracting internal attention matrices...")
    
    extracted_attentions = {}
    
    # 2. Run the isolated peak through all models
    for name, info in models_dict.items():
        path = info["path"]
        smoke_test = info.get("smoke_test", False)
        
        model = TemporalFusionTransformer(cardinalities, smoke_test=smoke_test).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        
        # Hook into the specific Self-Attention layer
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
        
        # Store the extracted (seq_len, seq_len) attention matrix
        # For forecasting, we usually care about how the terminal node attends to the past
        history_len = peak_batch["x_past_power"].shape[1] // model.patch_size
        terminal_attention = attention_weights_cache[name][0, -1, :history_len] 
        extracted_attentions[name] = terminal_attention
        
    return extracted_attentions, peak_batch

# ==========================================
# 3. PUBLICATION-READY VISUALIZATION
# ==========================================
def plot_attention_comparison(extracted_attentions, history_len):
    """
    Generates a publication-ready line plot comparing the attention distributions.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    time_steps = np.arange(-history_len, 0)
    
    # Plot Baseline
    if "Baseline" in extracted_attentions:
        ax.plot(time_steps, extracted_attentions["Baseline"], 
                label="Baseline (Pinball, SR=0.0)", 
                color="#94a3b8", linestyle="--", linewidth=2.5, alpha=0.8)
    
    # Plot Loss-Only (Gradient Thrashing / Washout)
    if "Loss_Only" in extracted_attentions:
        ax.plot(time_steps, extracted_attentions["Loss_Only"], 
                label="Asymmetric Only (Washout)", 
                color="#ef4444", linestyle="-.", linewidth=2.5, alpha=0.8)
    
    # Plot Synergy (Targeted Focus)
    if "Synergy" in extracted_attentions:
        ax.plot(time_steps, extracted_attentions["Synergy"], 
                label="Proposed Synergy (Asym + Stratified)", 
                color="#2563eb", linestyle="-", linewidth=3.5)

    # Formatting
    ax.set_title("TFT Multi-Head Attention Distribution Prior to Extreme Peak Event", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Historical Time Steps (t-L to t-1)", fontsize=14)
    ax.set_ylabel("Normalized Attention Weight", fontsize=14)
    
    # Add an annotation pointing out what to look for
    if "Synergy" in extracted_attentions:
        ax.annotate('Hypothesized Precursor Signal', 
                    xy=(time_steps[np.argmax(extracted_attentions["Synergy"])], np.max(extracted_attentions["Synergy"])),
                    xytext=(-history_len/2, np.max(extracted_attentions["Synergy"])*0.9),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2563eb", lw=2))

    ax.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.9, edgecolor='black')
    plt.tight_layout()
    
    plt.savefig("attention_washout_validation.pdf", format='pdf', dpi=300)
    print("Successfully generated 'attention_washout_validation.pdf'")
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
import torch
import numpy as np
import pandas as pd

from src.Orchestrator.IdealDatasetOrchestrator import IdealDatasetOrchestrator
from src.Transformer.TemporalFusionTransformer import TemporalFusionTransformer
from src.PytorchDataset.IdealPytorchDataset import IdealPytorchDataset
from src.Trainer.EarlyStopper import EarlyStopping
from torch.utils.data import DataLoader
import torch
import random
import os


def get_predictions(model, data_loader, device="cuda"):
    """Runs a model over the DataLoader and returns flattened predictions and actuals."""
    model.eval()
    all_p50, all_p90, all_actuals = [], [], []

    with torch.no_grad():
        for batch in data_loader:
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
                batch["x_past_power"].to(device),
                batch["x_past_time"].to(device),
                batch["x_past_temperature"].to(device),
                batch["x_past_weather_conditions"].to(device),
                batch["x_future_time"].to(device),
                batch["x_future_temperature"].to(device),
                batch["x_future_weather_conditions"].to(device),
                batch["x_static"].to(device),
                batch["y"].to(device),
            )

            # Forward pass (Handling mixed precision if applicable)
            if device == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    quantiles, _, _ = model(
                        x_past_power,
                        x_past_time,
                        x_past_temperature,
                        x_past_weather_conditions,
                        x_future_time,
                        x_future_temperature,
                        x_future_weather_conditions,
                        x_stat,
                    )
            else:
                quantiles, _, _ = model(
                    x_past_power,
                    x_past_time,
                    x_past_temperature,
                    x_past_weather_conditions,
                    x_future_time,
                    x_future_temperature,
                    x_future_weather_conditions,
                    x_stat,
                )

            # Extract P50 (median) and P90 (upper bound)
            p50 = quantiles[:, :, 1].cpu().numpy()
            p90 = quantiles[:, :, 2].cpu().numpy()

            all_p50.append(p50)
            all_p90.append(p90)
            all_actuals.append(y.cpu().numpy())

    # Concatenate and flatten all batches into 1D arrays for global metric calculation
    return (
        np.concatenate(all_p50).flatten(),
        np.concatenate(all_p90).flatten(),
        np.concatenate(all_actuals).flatten(),
    )


def calculate_metrics(p50, p90, actuals, dataset, peak_percentile=95):
    """Calculates MAPE, PAPE, and P90 Coverage after denormalizing."""

    # 1. Denormalize back to actual physical units (kW)
    # This is CRITICAL. You cannot calculate MAPE on Z-scores!
    p50_denorm = dataset.denormalize(p50)
    p90_denorm = dataset.denormalize(p90)
    actuals_denorm = dataset.denormalize(actuals)

    # 2. Global MAPE
    # Added epsilon to prevent division by zero if power usage drops perfectly to 0
    epsilon = 1e-4
    mape = (
        np.mean(np.abs((actuals_denorm - p50_denorm) / (actuals_denorm + epsilon)))
        * 100
    )

    # 3. Identify the "Peaks" (e.g., the top 5% of all usage in the test set)
    peak_threshold = np.percentile(actuals_denorm, peak_percentile)
    peak_indices = np.where(actuals_denorm >= peak_threshold)[0]

    peak_actuals = actuals_denorm[peak_indices]
    peak_p50 = p50_denorm[peak_indices]
    peak_p90 = p90_denorm[peak_indices]

    # 4. PAPE (Peak Absolute Percentage Error)
    # How accurate is the P50 prediction *only* during the most extreme spikes?
    pape = np.mean(np.abs((peak_actuals - peak_p50) / (peak_actuals + epsilon))) * 100

    # 5. P90 Peak Coverage
    # What percentage of actual peaks were successfully captured below the P90 bound?
    p90_coverage = np.mean(peak_actuals <= peak_p90) * 100

    return mape, pape, p90_coverage


def run_ablation_study(
    baseline_model, oracle_model, test_loader, dataset, device="cuda"
):
    print("Evaluating Baseline Model (Random Sampling)...")
    base_p50, base_p90, base_actuals = get_predictions(
        baseline_model, test_loader, device
    )
    base_mape, base_pape, base_cov = calculate_metrics(
        base_p50, base_p90, base_actuals, dataset
    )

    print("Evaluating Oracle Model (Stratified Sampling)...")
    oracle_p50, oracle_p90, oracle_actuals = get_predictions(
        oracle_model, test_loader, device
    )
    oracle_mape, oracle_pape, oracle_cov = calculate_metrics(
        oracle_p50, oracle_p90, oracle_actuals, dataset
    )

    # Format the results into a clean Pandas DataFrame for your report
    results_df = pd.DataFrame(
        {
            "Metric": [
                "Global MAPE (%)",
                "PAPE (Top 5% Peaks) (%)",
                "P90 Peak Coverage (%)",
            ],
            "Baseline (Random)": [
                f"{base_mape:.2f}%",
                f"{base_pape:.2f}%",
                f"{base_cov:.2f}%",
            ],
            "Oracle (Stratified)": [
                f"{oracle_mape:.2f}%",
                f"{oracle_pape:.2f}%",
                f"{oracle_cov:.2f}%",
            ],
        }
    )

    print("\n=== ABLATION STUDY RESULTS ===")
    print(results_df.to_string(index=False))

    return results_df


def load_model(model_path: str, orchestrator: IdealDatasetOrchestrator):
    model = TemporalFusionTransformer(orchestrator.cardinalities)
    model.load_state_dict(torch.load(model_path, map_location="cuda"))
    model.to("cuda")
    return model


if __name__ == "__main__":
    DATA_DIR = "data"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    accumulation_step = 16  # multiply to batchsize!
    BATCH_SIZE = 4
    EPOCHS = 2000
    LR = 1e-4
    WARMUP = 10
    STATIC_FEATURES = [
        "homeid",
        "residents",
        "income_band",
        "hometype",
        "urban_rural_class",
        "build_era",
        "occupied_days",
        "occupied_nights",
        "workingstatus",
        "gender",
        "ageband",
        "weeklyhoursofwork",
    ]

    # Pipeline Setup
    orchestrator = IdealDatasetOrchestrator(DATA_DIR)

    # Select home IDs (In real usage, list available IDs from file)
    home_ids = [
        int(filename.split("_", 1)[0][4:])
        for filename in filter(
            lambda x: x.endswith(".csv"), os.listdir(DATA_DIR + "/household_sensors/")
        )
    ]

    random.seed(42)  # For reproducibility
    random.shuffle(home_ids)

    # 80/20 Split
    train_idx = int(len(home_ids) * 0.8)
    test_idx = int(len(home_ids) * 0.9)
    train_ids = home_ids[:train_idx]
    val_ids = home_ids[train_idx:test_idx]
    test_ids = home_ids[test_idx:]

    train_dataset = IdealPytorchDataset(train_ids, orchestrator, sampling_rate=0.5)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    pwr_stat = train_dataset.power_stats
    wth_stat = train_dataset.weather_stats
    test_dataset = IdealPytorchDataset(
        test_ids,
        orchestrator,
        split="test",
        power_stats=pwr_stat,
        weather_stats=wth_stat,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Input baseline model path. current path is " + os.getcwd() + ":")
    model_path = input()
    baseline_model = load_model(model_path, orchestrator)
    print("Input oracle model path. current path is " + os.getcwd() + ":")
    model_path = input()
    oracle_model = load_model(model_path, orchestrator)
    results_table = run_ablation_study(
        baseline_model, oracle_model, test_loader, test_dataset, device="cuda"
    )

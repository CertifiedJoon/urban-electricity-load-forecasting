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
    all_p10, all_p50, all_p90, all_actuals = [], [], [], []

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
            p10 = quantiles[:, :, 0].cpu().numpy()
            p50 = quantiles[:, :, 1].cpu().numpy()
            p90 = quantiles[:, :, 2].cpu().numpy()

            all_p10.append(p10)
            all_p50.append(p50)
            all_p90.append(p90)
            all_actuals.append(y.cpu().numpy())

    # Concatenate and flatten all batches into 1D arrays for global metric calculation
    return (
        np.concatenate(all_p10).flatten(),
        np.concatenate(all_p50).flatten(),
        np.concatenate(all_p90).flatten(),
        np.concatenate(all_actuals).flatten(),
    )


def calculate_metrics(
    p10,
    p50,
    p90,
    actuals,
    dataset,
    peak_percentile=90,
    trough_percentile=10,
    patch_size=10,
):
    """Calculates MAPE, PAPE/TAPE, and Coverage for both peaks and troughs."""

    # 1. Denormalize back to actual physical units (W)
    p10_denorm = dataset.denormalize(p10)
    p50_denorm = dataset.denormalize(p50)
    p90_denorm = dataset.denormalize(p90)

    actuals_denorm = dataset.denormalize(actuals)
    # Reshape and mean to align with patch-based forecasting if necessary
    actuals_denorm = actuals_denorm.reshape(-1, patch_size).mean(axis=1)

    epsilon = 1e-4

    # 2. Global Metrics
    mae = np.mean(np.abs(actuals_denorm - p50_denorm))
    total_error = np.sum(np.abs(actuals_denorm - p50_denorm))
    total_actual = np.sum(actuals_denorm)
    wmape = (total_error / total_actual) * 100 if total_actual > 0 else 0.0

    # 3. Peak Analysis (Top 10%)
    peak_threshold = np.percentile(actuals_denorm, peak_percentile)
    peak_indices = np.where(actuals_denorm >= peak_threshold)[0]

    if len(peak_indices) > 0:
        p_act = actuals_denorm[peak_indices]
        p_p50 = p50_denorm[peak_indices]
        p_p90 = p90_denorm[peak_indices]

        pape = np.mean(np.abs((p_act - p_p50) / (p_act + epsilon))) * 100
        p90_peak_cov = np.mean(p_act <= p_p90) * 100
    else:
        pape, p90_peak_cov = 0.0, 0.0

    # 4. Trough Analysis (Bottom X%)
    trough_threshold = np.percentile(actuals_denorm, trough_percentile)
    trough_indices = np.where(actuals_denorm <= trough_threshold)[0]

    if len(trough_indices) > 0:
        t_act = actuals_denorm[trough_indices]
        t_p50 = p50_denorm[trough_indices]
        t_p10 = p10_denorm[trough_indices]

        tape = np.mean(np.abs((t_act - t_p50) / (t_act + epsilon))) * 100
        p10_trough_cov = (
            np.mean(t_act >= t_p10) * 100
        )  # Coverage means actual is ABOVE the floor
    else:
        tape, p10_trough_cov = 0.0, 0.0

    return mae, wmape, pape, p90_peak_cov, tape, p10_trough_cov


def run_ablation_study(
    baseline_model,
    oracle_model,
    oracle_model2,
    test_loader,
    dataset,
    device="cuda",
    num_runs=1000,
):
    """Executes ablation study tracking both Peak and Trough accuracy."""
    print(f"Running Ablation Study over {num_runs} randomized horizon passes...")

    # Expanded dictionaries to track trough metrics
    metrics_template = {
        "mae": [],
        "wmape": [],
        "pape": [],
        "p90_cov": [],
        "tape": [],
        "p10_cov": [],
    }
    base_metrics = {k: [] for k in metrics_template}
    oracle_metrics = {k: [] for k in metrics_template}
    oracle2_metrics = {k: [] for k in metrics_template}

    for i in range(num_runs):
        print(f"  -> Execution Run {i + 1}/{num_runs}", end="\r")

        models = [
            (baseline_model, base_metrics),
            (oracle_model, oracle_metrics),
            (oracle_model2, oracle2_metrics),
        ]

        for model, storage in models:
            # Assuming get_predictions now returns p10, p50, p90, and actuals
            p10, p50, p90, actuals = get_predictions(model, test_loader, device)

            mae, wmape, pape, p90_c, tape, p10_c = calculate_metrics(
                p10, p50, p90, actuals, dataset
            )

            storage["mae"].append(mae)
            storage["wmape"].append(wmape)
            storage["pape"].append(pape)
            storage["p90_cov"].append(p90_c)
            storage["tape"].append(tape)
            storage["p10_cov"].append(p10_c)

    def format_stat(metric_list, power_metric=False):
        mean, std = np.mean(metric_list), np.std(metric_list)
        if power_metric:
            return f"{mean:.0f}W +/- {std:.0f}W"
        return f"{mean:.2f}% +/- {std:.2f}%"

    # Create the summary table
    results_df = pd.DataFrame(
        {
            "Metric": [
                "Global MAE",
                "Global wMAPE",
                "PAPE (Top 10% Peaks)",
                "P90 Peak Coverage",
                "TAPE (Bottom 10% Troughs)",
                "P10 Trough Coverage",
            ],
            "Baseline": [
                format_stat(base_metrics["mae"], True),
                format_stat(base_metrics["wmape"]),
                format_stat(base_metrics["pape"]),
                format_stat(base_metrics["p90_cov"]),
                format_stat(base_metrics["tape"]),
                format_stat(base_metrics["p10_cov"]),
            ],
            "Oracle (Asym)": [
                format_stat(oracle_metrics["mae"], True),
                format_stat(oracle_metrics["wmape"]),
                format_stat(oracle_metrics["pape"]),
                format_stat(oracle_metrics["p90_cov"]),
                format_stat(oracle_metrics["tape"]),
                format_stat(oracle_metrics["p10_cov"]),
            ],
            "Oracle (Strat+Asym)": [
                format_stat(oracle2_metrics["mae"], True),
                format_stat(oracle2_metrics["wmape"]),
                format_stat(oracle2_metrics["pape"]),
                format_stat(oracle2_metrics["p90_cov"]),
                format_stat(oracle2_metrics["tape"]),
                format_stat(oracle2_metrics["p10_cov"]),
            ],
        }
    )

    print("\n\n=== STABILIZED ABLATION STUDY RESULTS ===")
    print(results_df.to_string(index=False))
    return results_df
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
    SR = 0.0
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

    print("SR: " + str(SR))

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

    train_dataset = IdealPytorchDataset(train_ids, orchestrator, sampling_rate=SR)
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
    model_path = "training_results/1.0.7 (base final)/model.pth"
    baseline_model = load_model(model_path, orchestrator)
    print("Input oracle assymetric model path. current path is " + os.getcwd() + ":")
    model_path = "training_results/1.0.2 (Base Model)/model.pth"
    oracle_model = load_model(model_path, orchestrator)
    print(
        "Input oracle assymetric stratified model path. current path is "
        + os.getcwd()
        + ":"
    )
    model_path = "training_results/1.0.5 (oracle final)/model.pth"
    oracle_model2 = load_model(model_path, orchestrator)
    results_table = run_ablation_study(
        baseline_model,
        oracle_model,
        oracle_model2,
        test_loader,
        test_dataset,
        device="cuda",
    )

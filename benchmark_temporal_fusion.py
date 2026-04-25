import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.PytorchDataset.IdealPytorchDataset import IdealPytorchDataset
from src.Trainer.TemporalFusionTrainer import AsymmetricSpikeQuantileLoss
from train_temporal_fusion import create_data_loaders, run_training_experiment


VARIANTS = [
    {
        "name": "base_pinball_sr0.0",
        "sampling_rate": 0.0,
        "loss_name": "pinball",
    },
    {
        "name": "oracle_pinball_sr0.5",
        "sampling_rate": 0.5,
        "loss_name": "pinball",
    },
    {
        "name": "oracle_asymmetric_sr0.0",
        "sampling_rate": 0.0,
        "w_peak": 5,
        "w_trough": 5,
        "loss_name": "asymmetric",
    },
    {
        "name": "oracle_asymmetric_sr0.5",
        "sampling_rate": 0.5,
        "w_peak": 5,
        "w_trough": 5,
        "loss_name": "asymmetric",
    },
]

METRIC_PLOT_CONFIG = [
    ("mae_w", "MAE (W)", False),
    ("wmape_pct", "wMAPE (%)", False),
    ("pape_pct", "Peak APE (%)", False),
    ("tape_pct", "Trough APE (%)", False),
    ("p90_peak_coverage_pct", "P90 Peak Coverage (%)", True),
    ("p10_trough_coverage_pct", "P10 Trough Coverage (%)", True),
]


def compute_axis_limits(values, include_zero=False, reference_values=None, symmetric=False):
    numeric_values = np.asarray(values, dtype=float)
    numeric_values = numeric_values[np.isfinite(numeric_values)]

    extras = []
    if reference_values is not None:
        extras = np.atleast_1d(reference_values).astype(float).tolist()

    if numeric_values.size == 0 and not extras:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)

    combined = numeric_values
    if extras:
        combined = np.concatenate([numeric_values, np.asarray(extras, dtype=float)])

    if symmetric:
        max_abs = float(np.max(np.abs(combined))) if combined.size else 1.0
        pad = max(max_abs * 0.15, 1.0 if max_abs >= 1.0 else 0.05)
        return -max_abs - pad, max_abs + pad

    lower = float(np.min(combined)) if combined.size else 0.0
    upper = float(np.max(combined)) if combined.size else 1.0

    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)

    spread = upper - lower
    if spread == 0:
        pad = max(abs(upper) * 0.08, 1.0 if abs(upper) >= 1.0 else 0.05)
    else:
        pad = max(spread * 0.15, 1.0 if max(abs(lower), abs(upper)) >= 1.0 else 0.05)

    return lower - pad, upper + pad


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and benchmark multiple Temporal Fusion Transformer variants."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output-root", default=os.path.join("training_results", "benchmark_tft")
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=290)
    parser.add_argument("--accumulation-step", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-eval-samples-per-home",
        type=int,
        default=1,
        help="Deterministic windows per home for validation during training.",
    )
    parser.add_argument(
        "--test-eval-samples-per-home",
        type=int,
        default=8,
        help="Deterministic windows per home for final benchmark evaluation.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def make_test_loader(experiment, batch_size, device, eval_samples_per_home):
    benchmark_test_dataset = IdealPytorchDataset(
        experiment["test_ids"],
        experiment["orchestrator"],
        split="test",
        power_stats=experiment["train_dataset"].power_stats,
        weather_stats=experiment["train_dataset"].weather_stats,
        eval_samples_per_home=eval_samples_per_home,
    )
    _, _, test_loader = create_data_loaders(
        experiment["train_dataset"],
        experiment["val_dataset"],
        benchmark_test_dataset,
        batch_size=batch_size,
        device=device,
    )
    return benchmark_test_dataset, test_loader


def get_quantiles(model, batch, device):
    model_inputs = (
        batch["x_past_power"].to(device),
        batch["x_past_time"].to(device),
        batch["x_past_temperature"].to(device),
        batch["x_past_weather_conditions"].to(device),
        batch["x_future_time"].to(device),
        batch["x_future_temperature"].to(device),
        batch["x_future_weather_conditions"].to(device),
        batch["x_static"].to(device),
    )
    if str(device).startswith("cuda"):
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(*model_inputs)
    else:
        outputs = model(*model_inputs)

    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def collect_predictions(
    model, data_loader, device, baseline_loss, w_peak=2.0, w_trough=2.0
):
    model.eval()
    pinball_criterion = AsymmetricSpikeQuantileLoss(baseline=True)
    objective_criterion = AsymmetricSpikeQuantileLoss(
        baseline=baseline_loss,
        w_peak=w_peak,
        w_trough=w_trough,
    )

    all_quantiles = []
    all_targets = []
    pinball_losses = []
    objective_losses = []

    with torch.no_grad():
        for batch in data_loader:
            quantiles = get_quantiles(model, batch, device)
            targets = batch["y"].to(device)

            pinball_losses.append(pinball_criterion(quantiles, targets).item())
            objective_losses.append(objective_criterion(quantiles, targets).item())

            patched_targets = targets.view(targets.size(0), -1, 10).mean(dim=-1)
            all_quantiles.append(quantiles.cpu().numpy())
            all_targets.append(patched_targets.cpu().numpy())

    return {
        "quantiles": np.concatenate(all_quantiles, axis=0),
        "targets": np.concatenate(all_targets, axis=0),
        "pinball_loss": float(np.mean(pinball_losses)),
        "objective_loss": float(np.mean(objective_losses)),
    }


def evaluate_predictions(quantiles, patched_targets, dataset):
    p10 = dataset.denormalize(quantiles[:, :, 0].reshape(-1))
    p50 = dataset.denormalize(quantiles[:, :, 1].reshape(-1))
    p90 = dataset.denormalize(quantiles[:, :, 2].reshape(-1))
    actuals = dataset.denormalize(patched_targets.reshape(-1))

    epsilon = 1e-4
    mae = float(np.mean(np.abs(actuals - p50)))
    wmape = float((np.sum(np.abs(actuals - p50)) / np.sum(actuals + epsilon)) * 100)

    peak_threshold = np.percentile(actuals, 90)
    trough_threshold = np.percentile(actuals, 10)

    peak_mask = actuals >= peak_threshold
    trough_mask = actuals <= trough_threshold

    pape = (
        float(
            np.mean(
                np.abs(
                    (actuals[peak_mask] - p50[peak_mask])
                    / (actuals[peak_mask] + epsilon)
                )
            )
            * 100
        )
        if np.any(peak_mask)
        else 0.0
    )
    p90_peak_cov = (
        float(np.mean(actuals[peak_mask] <= p90[peak_mask]) * 100)
        if np.any(peak_mask)
        else 0.0
    )

    tape = (
        float(
            np.mean(
                np.abs(
                    (actuals[trough_mask] - p50[trough_mask])
                    / (actuals[trough_mask] + epsilon)
                )
            )
            * 100
        )
        if np.any(trough_mask)
        else 0.0
    )
    p10_trough_cov = (
        float(np.mean(actuals[trough_mask] >= p10[trough_mask]) * 100)
        if np.any(trough_mask)
        else 0.0
    )

    return {
        "mae_w": mae,
        "wmape_pct": wmape,
        "pape_pct": pape,
        "p90_peak_coverage_pct": p90_peak_cov,
        "tape_pct": tape,
        "p10_trough_coverage_pct": p10_trough_cov,
    }


def build_peak_trough_breakdown(quantiles, patched_targets, dataset):
    p10 = dataset.denormalize(quantiles[:, :, 0].reshape(-1))
    p50 = dataset.denormalize(quantiles[:, :, 1].reshape(-1))
    p90 = dataset.denormalize(quantiles[:, :, 2].reshape(-1))
    actuals = dataset.denormalize(patched_targets.reshape(-1))

    errors = p50 - actuals
    interval_width = p90 - p10

    peak_threshold = np.percentile(actuals, 90)
    trough_threshold = np.percentile(actuals, 10)
    peak_mask = actuals >= peak_threshold
    trough_mask = actuals <= trough_threshold

    def summarize(mask):
        if not np.any(mask):
            return {
                "mae_w": 0.0,
                "bias_w": 0.0,
                "interval_width_w": 0.0,
                "count": 0,
            }
        return {
            "mae_w": float(np.mean(np.abs(errors[mask]))),
            "bias_w": float(np.mean(errors[mask])),
            "interval_width_w": float(np.mean(interval_width[mask])),
            "count": int(np.sum(mask)),
        }

    return {
        "peak": summarize(peak_mask),
        "trough": summarize(trough_mask),
    }


def build_preview(quantiles, patched_targets, dataset, preview_points=144):
    p10 = dataset.denormalize(quantiles[:, :, 0].reshape(-1))
    p50 = dataset.denormalize(quantiles[:, :, 1].reshape(-1))
    p90 = dataset.denormalize(quantiles[:, :, 2].reshape(-1))
    actuals = dataset.denormalize(patched_targets.reshape(-1))

    limit = min(preview_points, len(actuals))
    return {
        "actuals": actuals[:limit].tolist(),
        "p10": p10[:limit].tolist(),
        "p50": p50[:limit].tolist(),
        "p90": p90[:limit].tolist(),
    }


def plot_metric_dashboard(results_df, output_root):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]

    for idx, (metric_key, title, target_coverage) in enumerate(METRIC_PLOT_CONFIG):
        ax = axes[idx]
        values = results_df[metric_key].to_numpy()
        bars = ax.bar(results_df["variant"], values, color=colors[: len(values)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)

        reference = None
        if target_coverage:
            reference = 90.0
            ax.axhline(
                reference,
                color="#6c757d",
                linestyle="--",
                linewidth=1,
            )

        lower, upper = compute_axis_limits(values, reference_values=reference)
        ax.set_ylim(lower, upper)

        for bar, value in zip(bars, values):
            label = (
                f"{value:,.0f}"
                if metric_key.endswith("_w")
                else f"{value:.1f}"
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Temporal Fusion Benchmark Dashboard", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = os.path.join(output_root, "benchmark_dashboard.png")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_forecast_preview(previews, output_root):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, variant_name in zip(axes, [variant["name"] for variant in VARIANTS]):
        preview = previews[variant_name]
        time_axis = np.arange(len(preview["actuals"]))

        ax.plot(
            time_axis,
            preview["actuals"],
            color="#111111",
            linewidth=1.5,
            label="Actual",
        )
        ax.plot(time_axis, preview["p50"], color="#1d3557", linewidth=1.5, label="P50")
        ax.fill_between(
            time_axis,
            preview["p10"],
            preview["p90"],
            color="#457b9d",
            alpha=0.25,
            label="P10-P90",
        )
        ax.set_title(variant_name)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Patch Index")
        ax.set_ylabel("Power (W)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Forecast Preview Across Variants", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(output_root, "forecast_preview.png")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_peak_trough_breakdown(breakdowns, output_root):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    variant_names = [variant["name"] for variant in VARIANTS]
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]

    plot_specs = [
        ("peak", "mae_w", "Peak MAE (W)"),
        ("peak", "bias_w", "Peak Bias (W)"),
        ("trough", "mae_w", "Trough MAE (W)"),
        ("trough", "bias_w", "Trough Bias (W)"),
    ]

    for ax, (bucket, metric_key, title) in zip(axes, plot_specs):
        values = [breakdowns[name][bucket][metric_key] for name in variant_names]
        bars = ax.bar(variant_names, values, color=colors[: len(values)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)
        if metric_key == "bias_w":
            ax.axhline(0.0, color="#6c757d", linestyle="--", linewidth=1)

        if metric_key == "bias_w":
            lower, upper = compute_axis_limits(values, reference_values=0.0, symmetric=True)
        else:
            lower, upper = compute_axis_limits(values)
        ax.set_ylim(lower, upper)

        for bar, value in zip(bars, values):
            y = bar.get_height()
            if metric_key == "bias_w" and value < 0:
                va = "top"
            else:
                va = "bottom"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{value:,.0f}",
                ha="center",
                va=va,
                fontsize=9,
            )

    peak_count = breakdowns[variant_names[0]]["peak"]["count"] if variant_names else 0
    trough_count = (
        breakdowns[variant_names[0]]["trough"]["count"] if variant_names else 0
    )
    fig.suptitle(
        f"Peak/Trough Error Breakdown (top 10% peaks n={peak_count}, bottom 10% troughs n={trough_count})",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(output_root, "peak_trough_breakdown.png")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def run_variant(base_config, variant):
    run_dir = os.path.join(base_config["output_root"], variant["name"])
    experiment = run_training_experiment(
        {
            "data_dir": base_config["data_dir"],
            "device": base_config["device"],
            "sampling_rate": variant["sampling_rate"],
            "loss_name": variant["loss_name"],
            "batch_size": base_config["batch_size"],
            "epochs": base_config["epochs"],
            "lr": base_config["lr"],
            "warmup_epochs": base_config["warmup_epochs"],
            "patience": base_config["patience"],
            "accumulation_step": base_config["accumulation_step"],
            "seed": base_config["seed"],
            "eval_samples_per_home": base_config["val_eval_samples_per_home"],
            "output_dir": run_dir,
            "smoke_test": base_config["smoke_test"],
        }
    )

    benchmark_dataset, benchmark_loader = make_test_loader(
        experiment=experiment,
        batch_size=base_config["batch_size"],
        device=base_config["device"],
        eval_samples_per_home=base_config["test_eval_samples_per_home"],
    )
    predictions = collect_predictions(
        experiment["model"],
        benchmark_loader,
        base_config["device"],
        baseline_loss=variant["loss_name"] == "pinball",
        w_peak=variant.get("w_peak", 2.0),
        w_trough=variant.get("w_trough", 2.0),
    )
    metrics = evaluate_predictions(
        predictions["quantiles"],
        predictions["targets"],
        benchmark_dataset,
    )
    breakdown = build_peak_trough_breakdown(
        predictions["quantiles"],
        predictions["targets"],
        benchmark_dataset,
    )
    preview = build_preview(
        predictions["quantiles"],
        predictions["targets"],
        benchmark_dataset,
    )
    metrics["pinball_loss"] = predictions["pinball_loss"]
    metrics["objective_loss"] = predictions["objective_loss"]
    metrics["variant"] = variant["name"]
    metrics["sampling_rate"] = variant["sampling_rate"]
    metrics["loss_name"] = variant["loss_name"]
    metrics["w_peak"] = variant.get("w_peak", 2.0)
    metrics["w_trough"] = variant.get("w_trough", 2.0)
    metrics["model_path"] = experiment["model_path"]
    return metrics, preview, breakdown


def save_benchmark_outputs(results, previews, breakdowns, output_root):
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(output_root, "benchmark_results.csv")
    results_json = os.path.join(output_root, "benchmark_results.json")
    dashboard_png = plot_metric_dashboard(results_df, output_root)
    preview_png = plot_forecast_preview(previews, output_root)
    breakdown_png = plot_peak_trough_breakdown(breakdowns, output_root)

    results_df.to_csv(results_csv, index=False)
    with open(results_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    return {
        "results_df": results_df,
        "results_csv": results_csv,
        "results_json": results_json,
        "dashboard_png": dashboard_png,
        "preview_png": preview_png,
        "breakdown_png": breakdown_png,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    base_config = {
        "data_dir": args.data_dir,
        "output_root": args.output_root,
        "device": args.device,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "accumulation_step": args.accumulation_step,
        "seed": args.seed,
        "val_eval_samples_per_home": args.val_eval_samples_per_home,
        "test_eval_samples_per_home": args.test_eval_samples_per_home,
        "smoke_test": args.smoke_test,
    }

    results = []
    previews = {}
    breakdowns = {}
    for variant in VARIANTS:
        print(f"\n=== Running {variant['name']} ===")
        metrics, preview, breakdown = run_variant(base_config, variant)
        results.append(metrics)
        previews[variant["name"]] = preview
        breakdowns[variant["name"]] = breakdown

    outputs = save_benchmark_outputs(results, previews, breakdowns, args.output_root)
    results_df = outputs["results_df"]

    print("\nBenchmark summary:")
    print(results_df.to_string(index=False))
    print(f"\nSaved CSV to {outputs['results_csv']}")
    print(f"Saved JSON to {outputs['results_json']}")
    print(f"Saved dashboard to {outputs['dashboard_png']}")
    print(f"Saved forecast preview to {outputs['preview_png']}")
    print(f"Saved peak/trough breakdown to {outputs['breakdown_png']}")


if __name__ == "__main__":
    main()

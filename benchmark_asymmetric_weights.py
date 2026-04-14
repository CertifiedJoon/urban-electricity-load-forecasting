import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from benchmark_temporal_fusion import (
    build_peak_trough_breakdown,
    collect_predictions,
    evaluate_predictions,
    make_test_loader,
)
from train_temporal_fusion import run_training_experiment


SWEEP_GROUPS = {
    "trough_only": [(0, w) for w in range(0, 21, 5)],
    "peak_only": [(w, 0) for w in range(0, 21, 5)],
    "diagonal": [(w, w) for w in range(0, 21, 5)],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark independent peak/trough weights for AsymmetricSpikeQuantileLoss."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join("training_results", "benchmark_asymmetric_weights"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--sampling-rate", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--accumulation-step", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-eval-samples-per-home", type=int, default=1)
    parser.add_argument("--test-eval-samples-per-home", type=int, default=8)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def build_unique_configs():
    unique = {}
    for group_name, pairs in SWEEP_GROUPS.items():
        for w_peak, w_trough in pairs:
            key = (w_peak, w_trough)
            if key not in unique:
                unique[key] = {
                    "w_peak": w_peak,
                    "w_trough": w_trough,
                    "config_name": f"wpeak_{w_peak}_wtrough_{w_trough}",
                    "groups": [],
                }
            unique[key]["groups"].append(group_name)
    return list(unique.values())


def explode_group_rows(results):
    rows = []
    for result in results:
        for group_name in result["groups"]:
            if group_name == "trough_only":
                sweep_value = result["w_trough"]
            elif group_name == "peak_only":
                sweep_value = result["w_peak"]
            else:
                sweep_value = result["w_peak"]

            row = dict(result)
            row["group_name"] = group_name
            row["sweep_value"] = sweep_value
            rows.append(row)
    return pd.DataFrame(rows)


def plot_group_metric_sweeps(group_df, output_root):
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex="col")
    group_order = ["trough_only", "peak_only", "diagonal"]
    group_titles = {
        "trough_only": "W_peak = 0, sweep W_trough",
        "peak_only": "W_trough = 0, sweep W_peak",
        "diagonal": "Diagonal sweep: W_peak = W_trough",
    }
    colors = {
        "mae_kw": "#264653",
        "pape_pct": "#e76f51",
        "tape_pct": "#2a9d8f",
        "p90_peak_coverage_pct": "#1d3557",
        "p10_trough_coverage_pct": "#8d99ae",
        "objective_loss": "#e9c46a",
    }

    for row_idx, group_name in enumerate(group_order):
        subset = group_df[group_df["group_name"] == group_name].sort_values("sweep_value")
        x = subset["sweep_value"].to_numpy()

        error_ax = axes[row_idx, 0]
        error_ax.plot(x, subset["mae_kw"], marker="o", color=colors["mae_kw"], label="MAE (kW)")
        error_ax.plot(x, subset["pape_pct"], marker="o", color=colors["pape_pct"], label="PAPE (%)")
        error_ax.plot(x, subset["tape_pct"], marker="o", color=colors["tape_pct"], label="TAPE (%)")
        error_ax.set_title(group_titles[group_name])
        error_ax.set_ylabel("Error")
        error_ax.grid(alpha=0.2)
        if row_idx == 0:
            error_ax.legend(frameon=False)

        coverage_ax = axes[row_idx, 1]
        coverage_ax.plot(
            x,
            subset["p90_peak_coverage_pct"],
            marker="o",
            color=colors["p90_peak_coverage_pct"],
            label="P90 Peak Cov (%)",
        )
        coverage_ax.plot(
            x,
            subset["p10_trough_coverage_pct"],
            marker="o",
            color=colors["p10_trough_coverage_pct"],
            label="P10 Trough Cov (%)",
        )
        coverage_ax.plot(
            x,
            subset["objective_loss"],
            marker="o",
            color=colors["objective_loss"],
            label="Objective Loss",
        )
        coverage_ax.axhline(90, color="#6c757d", linestyle="--", linewidth=1)
        coverage_ax.set_ylabel("Coverage / Loss")
        coverage_ax.grid(alpha=0.2)
        if row_idx == 0:
            coverage_ax.legend(frameon=False)

    axes[-1, 0].set_xlabel("Sweep Weight Value")
    axes[-1, 1].set_xlabel("Sweep Weight Value")
    fig.suptitle("Asymmetric Weight Sweep Summary", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = os.path.join(output_root, "weight_sweep_summary.png")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_group_bias_breakdown(group_df, output_root):
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex="col")
    group_order = ["trough_only", "peak_only", "diagonal"]
    group_titles = {
        "trough_only": "W_peak = 0, sweep W_trough",
        "peak_only": "W_trough = 0, sweep W_peak",
        "diagonal": "Diagonal sweep: W_peak = W_trough",
    }

    for row_idx, group_name in enumerate(group_order):
        subset = group_df[group_df["group_name"] == group_name].sort_values("sweep_value")
        x = subset["sweep_value"].to_numpy()

        peak_ax = axes[row_idx, 0]
        peak_ax.plot(x, subset["peak_mae_kw"], marker="o", color="#e76f51", label="Peak MAE")
        peak_ax.plot(x, subset["peak_bias_kw"], marker="o", color="#d62828", label="Peak Bias")
        peak_ax.plot(
            x,
            subset["peak_interval_width_kw"],
            marker="o",
            color="#f4a261",
            label="Peak Interval Width",
        )
        peak_ax.axhline(0, color="#6c757d", linestyle="--", linewidth=1)
        peak_ax.set_title(group_titles[group_name])
        peak_ax.set_ylabel("Peak Metrics (kW)")
        peak_ax.grid(alpha=0.2)
        if row_idx == 0:
            peak_ax.legend(frameon=False)

        trough_ax = axes[row_idx, 1]
        trough_ax.plot(x, subset["trough_mae_kw"], marker="o", color="#2a9d8f", label="Trough MAE")
        trough_ax.plot(x, subset["trough_bias_kw"], marker="o", color="#1d3557", label="Trough Bias")
        trough_ax.plot(
            x,
            subset["trough_interval_width_kw"],
            marker="o",
            color="#8ecae6",
            label="Trough Interval Width",
        )
        trough_ax.axhline(0, color="#6c757d", linestyle="--", linewidth=1)
        trough_ax.set_ylabel("Trough Metrics (kW)")
        trough_ax.grid(alpha=0.2)
        if row_idx == 0:
            trough_ax.legend(frameon=False)

    axes[-1, 0].set_xlabel("Sweep Weight Value")
    axes[-1, 1].set_xlabel("Sweep Weight Value")
    fig.suptitle("Peak/Trough Bias and Interval Width Across Weight Sweeps", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = os.path.join(output_root, "weight_sweep_bias_breakdown.png")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def run_weight_config(base_config, config):
    run_dir = os.path.join(base_config["output_root"], config["config_name"])
    experiment = run_training_experiment(
        {
            "data_dir": base_config["data_dir"],
            "device": base_config["device"],
            "sampling_rate": base_config["sampling_rate"],
            "loss_name": "asymmetric",
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
            "w_peak": config["w_peak"],
            "w_trough": config["w_trough"],
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
        baseline_loss=False,
        w_peak=config["w_peak"],
        w_trough=config["w_trough"],
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

    result = {
        "config_name": config["config_name"],
        "groups": config["groups"],
        "w_peak": config["w_peak"],
        "w_trough": config["w_trough"],
        "sampling_rate": base_config["sampling_rate"],
        "objective_loss": predictions["objective_loss"],
        "pinball_loss": predictions["pinball_loss"],
        "model_path": experiment["model_path"],
    }
    result.update(metrics)
    result["peak_mae_kw"] = breakdown["peak"]["mae_kw"]
    result["peak_bias_kw"] = breakdown["peak"]["bias_kw"]
    result["peak_interval_width_kw"] = breakdown["peak"]["interval_width_kw"]
    result["peak_count"] = breakdown["peak"]["count"]
    result["trough_mae_kw"] = breakdown["trough"]["mae_kw"]
    result["trough_bias_kw"] = breakdown["trough"]["bias_kw"]
    result["trough_interval_width_kw"] = breakdown["trough"]["interval_width_kw"]
    result["trough_count"] = breakdown["trough"]["count"]
    return result


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    base_config = {
        "data_dir": args.data_dir,
        "output_root": args.output_root,
        "device": args.device,
        "sampling_rate": args.sampling_rate,
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

    unique_configs = build_unique_configs()
    results = []

    for config in unique_configs:
        print(
            f"\n=== Running {config['config_name']} "
            f"(W_peak={config['w_peak']}, W_trough={config['w_trough']}) ==="
        )
        results.append(run_weight_config(base_config, config))

    results_df = pd.DataFrame(results).sort_values(["w_peak", "w_trough"]).reset_index(drop=True)
    group_df = explode_group_rows(results)

    results_csv = os.path.join(args.output_root, "asymmetric_weight_results.csv")
    group_csv = os.path.join(args.output_root, "asymmetric_weight_group_results.csv")
    results_json = os.path.join(args.output_root, "asymmetric_weight_results.json")

    results_df.to_csv(results_csv, index=False)
    group_df.to_csv(group_csv, index=False)
    with open(results_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    summary_png = plot_group_metric_sweeps(group_df, args.output_root)
    bias_png = plot_group_bias_breakdown(group_df, args.output_root)

    print("\nUnique configuration summary:")
    print(
        results_df[
            [
                "config_name",
                "w_peak",
                "w_trough",
                "objective_loss",
                "mae_kw",
                "pape_pct",
                "tape_pct",
                "p90_peak_coverage_pct",
                "p10_trough_coverage_pct",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved unique results CSV to {results_csv}")
    print(f"Saved group results CSV to {group_csv}")
    print(f"Saved JSON to {results_json}")
    print(f"Saved summary plot to {summary_png}")
    print(f"Saved bias breakdown plot to {bias_png}")


if __name__ == "__main__":
    main()

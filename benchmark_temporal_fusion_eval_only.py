import argparse
import json
import os

import torch

from benchmark_temporal_fusion import (
    VARIANTS,
    build_peak_trough_breakdown,
    build_preview,
    collect_predictions,
    evaluate_predictions,
    make_test_loader,
    save_benchmark_outputs,
)
from train_temporal_fusion import build_datasets, get_home_ids, split_home_ids
from src.Orchestrator.IdealDatasetOrchestrator import IdealDatasetOrchestrator
from src.Transformer.TemporalFusionTransformer import TemporalFusionTransformer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reload saved TFT benchmark models and rerun evaluation only."
    )
    parser.add_argument(
        "--benchmark-root",
        default=os.path.join("training_results", "benchmark_tft"),
        help="Folder containing per-variant subdirectories with model.pth and run_config.json.",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-eval-samples-per-home", type=int, default=8)
    return parser.parse_args()


def load_run_config(run_dir):
    config_path = os.path.join(run_dir, "run_config.json")
    model_path = os.path.join(run_dir, "model.pth")
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        return None

    with open(config_path, "r", encoding="utf-8") as handle:
        run_config = json.load(handle)
    run_config["model_path"] = model_path
    run_config["run_dir"] = run_dir
    run_config["config_name"] = os.path.basename(run_dir.rstrip("/\\"))
    return run_config


def discover_saved_runs(benchmark_root):
    discovered = {}
    for entry in os.scandir(benchmark_root):
        if not entry.is_dir():
            continue
        run_config = load_run_config(entry.path)
        if run_config is None:
            continue
        discovered[entry.name] = run_config
    return discovered


def rebuild_experiment_context(data_dir, seed, sampling_rate, eval_samples_per_home):
    orchestrator = IdealDatasetOrchestrator(data_dir)
    home_ids = get_home_ids(data_dir)
    train_ids, val_ids, test_ids = split_home_ids(home_ids, seed=seed)
    train_dataset, val_dataset, test_dataset = build_datasets(
        orchestrator=orchestrator,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        sampling_rate=sampling_rate,
        eval_samples_per_home=eval_samples_per_home,
    )
    return {
        "orchestrator": orchestrator,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "test_ids": test_ids,
    }


def load_model(orchestrator, model_path, device, smoke_test=False):
    model = TemporalFusionTransformer(
        orchestrator.cardinalities,
        smoke_test=smoke_test,
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def main():
    args = parse_args()
    saved_runs = discover_saved_runs(args.benchmark_root)
    if not saved_runs:
        raise FileNotFoundError(
            f"No saved runs found under {args.benchmark_root}. Expected model folders with model.pth and run_config.json."
        )

    results = []
    previews = {}
    breakdowns = {}

    for variant in VARIANTS:
        variant_name = variant["name"]
        if variant_name not in saved_runs:
            print(f"Skipping missing run for {variant_name}.")
            continue

        run_config = saved_runs[variant_name]
        print(f"\n=== Reloading {variant_name} ===")

        context = rebuild_experiment_context(
            data_dir=args.data_dir,
            seed=run_config.get("seed", 42),
            sampling_rate=run_config.get("sampling_rate", variant.get("sampling_rate", 0.0)),
            eval_samples_per_home=run_config.get("eval_samples_per_home", 1),
        )
        model = load_model(
            orchestrator=context["orchestrator"],
            model_path=run_config["model_path"],
            device=args.device,
            smoke_test=run_config.get("smoke_test", False),
        )

        experiment = {
            "orchestrator": context["orchestrator"],
            "train_dataset": context["train_dataset"],
            "val_dataset": context["val_dataset"],
            "test_dataset": context["test_dataset"],
            "test_ids": context["test_ids"],
            "model": model,
            "model_path": run_config["model_path"],
        }
        benchmark_dataset, benchmark_loader = make_test_loader(
            experiment=experiment,
            batch_size=args.batch_size,
            device=args.device,
            eval_samples_per_home=args.test_eval_samples_per_home,
        )
        predictions = collect_predictions(
            model,
            benchmark_loader,
            args.device,
            baseline_loss=run_config.get("loss_name", variant.get("loss_name")) == "pinball",
            w_peak=run_config.get("w_peak", variant.get("w_peak", 2.0)),
            w_trough=run_config.get("w_trough", variant.get("w_trough", 2.0)),
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
        metrics["variant"] = variant_name
        metrics["sampling_rate"] = run_config.get("sampling_rate", variant.get("sampling_rate", 0.0))
        metrics["loss_name"] = run_config.get("loss_name", variant.get("loss_name", "pinball"))
        metrics["w_peak"] = run_config.get("w_peak", variant.get("w_peak", 2.0))
        metrics["w_trough"] = run_config.get("w_trough", variant.get("w_trough", 2.0))
        metrics["model_path"] = run_config["model_path"]

        results.append(metrics)
        previews[variant_name] = preview
        breakdowns[variant_name] = breakdown

    if not results:
        raise RuntimeError("No saved models were successfully re-evaluated.")

    outputs = save_benchmark_outputs(results, previews, breakdowns, args.benchmark_root)
    results_df = outputs["results_df"]

    print("\nReloaded benchmark summary:")
    print(results_df.to_string(index=False))
    print(f"\nSaved CSV to {outputs['results_csv']}")
    print(f"Saved JSON to {outputs['results_json']}")
    print(f"Saved dashboard to {outputs['dashboard_png']}")
    print(f"Saved forecast preview to {outputs['preview_png']}")
    print(f"Saved peak/trough breakdown to {outputs['breakdown_png']}")


if __name__ == "__main__":
    main()

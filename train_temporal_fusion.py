import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.Orchestrator.IdealDatasetOrchestrator import IdealDatasetOrchestrator
from src.PytorchDataset.IdealPytorchDataset import IdealPytorchDataset
from src.Trainer.EarlyStopper import EarlyStopping
from src.Trainer.TemporalFusionTrainer import TemporalFusionTrainer
from src.Transformer.TemporalFusionTransformer import TemporalFusionTransformer
from src.interpret import visualize_density_heatmap


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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_home_ids(data_dir):
    sensor_dir = os.path.join(data_dir, "household_sensors")
    home_ids = [
        int(filename.split("_", 1)[0][4:])
        for filename in os.listdir(sensor_dir)
        if filename.endswith(".csv")
    ]
    home_ids.sort()
    return home_ids


def split_home_ids(home_ids, seed=42):
    shuffled = list(home_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    train_idx = int(len(shuffled) * 0.8)
    val_idx = int(len(shuffled) * 0.9)

    return (
        shuffled[:train_idx],
        shuffled[train_idx:val_idx],
        shuffled[val_idx:],
    )


def build_datasets(
    orchestrator,
    train_ids,
    val_ids,
    test_ids,
    sampling_rate,
    eval_samples_per_home,
):
    train_dataset = IdealPytorchDataset(
        train_ids,
        orchestrator,
        split="train",
        sampling_rate=sampling_rate,
    )
    power_stats = train_dataset.power_stats
    weather_stats = train_dataset.weather_stats

    val_dataset = IdealPytorchDataset(
        val_ids,
        orchestrator,
        split="val",
        power_stats=power_stats,
        weather_stats=weather_stats,
        eval_samples_per_home=eval_samples_per_home,
    )
    test_dataset = IdealPytorchDataset(
        test_ids,
        orchestrator,
        split="test",
        power_stats=power_stats,
        weather_stats=weather_stats,
        eval_samples_per_home=eval_samples_per_home,
    )
    return train_dataset, val_dataset, test_dataset


def build_scheduler(optimizer, epochs, warmup_epochs):
    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_epochs = max(1, epochs - warmup_epochs)
    later_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, later_scheduler],
        milestones=[warmup_epochs],
    )


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size, device):
    use_cuda = str(device).startswith("cuda")
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": use_cuda,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def run_training_experiment(config):
    seed_everything(config["seed"])
    device = config["device"]
    os.makedirs(config["output_dir"], exist_ok=True)

    orchestrator = IdealDatasetOrchestrator(config["data_dir"])
    home_ids = get_home_ids(config["data_dir"])
    train_ids, val_ids, test_ids = split_home_ids(home_ids, seed=config["seed"])

    train_dataset, val_dataset, test_dataset = build_datasets(
        orchestrator=orchestrator,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        sampling_rate=config["sampling_rate"],
        eval_samples_per_home=config["eval_samples_per_home"],
    )
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=config["batch_size"],
        device=device,
    )

    model = TemporalFusionTransformer(
        orchestrator.cardinalities,
        smoke_test=config["smoke_test"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = build_scheduler(
        optimizer,
        epochs=config["epochs"],
        warmup_epochs=config["warmup_epochs"],
    )
    checkpoint_path = os.path.join(config["output_dir"], "model.pth")
    early_stopping = EarlyStopping(
        patience=config["patience"],
        verbose=True,
        save_path=checkpoint_path,
    )
    trainer = TemporalFusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        result_path=config["output_dir"],
        device=device,
        baseline=config["loss_name"] == "pinball",
    )

    run_metadata = {
        "data_dir": config["data_dir"],
        "sampling_rate": config["sampling_rate"],
        "loss_name": config["loss_name"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "accumulation_step": config["accumulation_step"],
        "lr": config["lr"],
        "warmup_epochs": config["warmup_epochs"],
        "patience": config["patience"],
        "seed": config["seed"],
        "eval_samples_per_home": config["eval_samples_per_home"],
        "device": device,
    }
    with open(os.path.join(config["output_dir"], "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)

    for epoch in range(config["epochs"]):
        train_loss = trainer.train_epoch(config["accumulation_step"])
        val_loss = trainer.validate(config["accumulation_step"])
        train_dataset.update_loss_trend(val_loss)

        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6g}"
        )

        scheduler.step()
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Training stopped.")
            break

    trainer.plot_learning_curves()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    return {
        "model": model,
        "model_path": checkpoint_path,
        "trainer": trainer,
        "orchestrator": orchestrator,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "output_dir": config["output_dir"],
        "config": run_metadata,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Temporal Fusion Transformer.")
    parser.add_argument("sampling_rate", nargs="?", type=float, default=0.0)
    parser.add_argument(
        "--loss",
        choices=["pinball", "asymmetric"],
        default="pinball",
        help="Training loss. 'pinball' uses standard quantile loss.",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--accumulation-step", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-samples-per-home", type=int, default=1)
    parser.add_argument("--skip-interpret", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            "training_results",
            f"tft_sr_{args.sampling_rate:g}_{args.loss}",
        )

    experiment = run_training_experiment(
        {
            "data_dir": args.data_dir,
            "device": args.device,
            "sampling_rate": args.sampling_rate,
            "loss_name": args.loss,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "warmup_epochs": args.warmup_epochs,
            "patience": args.patience,
            "accumulation_step": args.accumulation_step,
            "seed": args.seed,
            "eval_samples_per_home": args.eval_samples_per_home,
            "output_dir": output_dir,
            "smoke_test": args.smoke_test,
        }
    )

    if args.skip_interpret:
        return

    for test_id in experiment["test_ids"]:
        visualize_density_heatmap(
            experiment["model"],
            experiment["test_dataset"],
            test_id,
            path=experiment["output_dir"],
            device=args.device,
            smoke_test=args.smoke_test,
        )


if __name__ == "__main__":
    main()

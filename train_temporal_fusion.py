from src.Orchestrator.IdealDatasetOrchestrator import IdealDatasetOrchestrator
from src.Transformer.TemporalFusionTransformer import TemporalFusionTransformer
from src.PytorchDataset.IdealPytorchDataset import IdealPytorchDataset
from src.Trainer.TemporalFusionTrainer import TemporalFusionTrainer
from src.Trainer.EarlyStopper import EarlyStopping
from src.interpret import visualize_tft_rolling_week, visualize_density_heatmap
from torch.utils.data import DataLoader
import torch
import random
import os

# os.environ["ONEDNN_VERBOSE"] = "all"
if __name__ == "__main__":
    # Settings
    DATA_DIR = "data"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    accumulation_step = 16  # multiply to batchsize!
    BATCH_SIZE = 4
    EPOCHS = 500
    LR = 1e-6
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
    early_stopping = EarlyStopping(patience=100, verbose=True, save_path="model.pth")

    # Select home IDs (In real usage, list available IDs from file)
    home_ids = [
        int(filename.split("_", 1)[0][4:])
        for filename in filter(
            lambda x: x.endswith(".csv"), os.listdir(DATA_DIR + "/household_sensors/")
        )
    ]

    # choose mode
    print("1. Train + Interpret\n2. Interpret\n3. Smoke Test\nType 1 or 2 or 3:")
    choice = int(input())
    # choice = 1
    
    train_dataset = IdealPytorchDataset(home_ids, orchestrator)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    pwr_stat = train_dataset.power_stats
    wth_stat = train_dataset.weather_stats

    if choice == 1:
        val_dataset = IdealPytorchDataset(
            home_ids,
            orchestrator,
            split="val",
            power_stats=pwr_stat,
            weather_stats=wth_stat,
        )
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        test_dataset = IdealPytorchDataset(
            home_ids,
            orchestrator,
            split="test",
            power_stats=pwr_stat,
            weather_stats=wth_stat,
        )

        model = TemporalFusionTransformer(orchestrator.cardinalities)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=50
        )

        trainer = TemporalFusionTrainer(
            model, train_loader, val_loader, optimizer, scheduler, device=DEVICE
        )

        for epoch in range(EPOCHS):
            train_loss = trainer.train_epoch(accumulation_step)
            val_loss = trainer.validate(accumulation_step)
            train_dataset.update_loss_trend(train_loss)
            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']}"
            )
            scheduler.step(val_loss)
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping triggered. Training stopped.")
                break

        trainer.plot_learning_curves()
        # visualize_tft_rolling_week(
        #     model,
        #     val_dataset,
        #     home_ids[split_idx],
        #     feature_names=STATIC_FEATURES,
        #     device=DEVICE,
        # )
        test_ids = [sample['homeid'] for sample in test_dataset.samples]
        for test_id in test_ids:
            visualize_density_heatmap(model, test_dataset, test_id, device=DEVICE)
    elif choice == 2:
        test_dataset = IdealPytorchDataset(
            home_ids,
            orchestrator,
            split="test",
            power_stats=pwr_stat,
            weather_stats=wth_stat,
        )
        print("Input .pth path. current path is " + os.getcwd() + ":")
        model_path = input()
        model = TemporalFusionTransformer(orchestrator.cardinalities)
        model.load_state_dict(torch.load(model_path, map_location="cuda"))
        model.to("cuda")
        # visualize_tft_rolling_week(
        #     model,
        #     val_dataset,
        #     home_ids[split_idx],
        #     feature_names=STATIC_FEATURES,
        #     device=DEVICE,
        # )
        test_ids = [sample['homeid'] for sample in test_dataset.samples]
        for test_id in test_ids:
            visualize_density_heatmap(model, test_dataset, test_id, device=DEVICE)
    elif choice == 3:
        print("RUNNING IN SMOKE TEST MODE (CPU)")
        # Overwrite config for speed
        BATCH_SIZE = 2
        EPOCHS = 1
        MAX_BATCHES_PER_EPOCH = 5

        val_dataset = IdealPytorchDataset(
            home_ids,
            orchestrator,
            split="val",
            power_stats=pwr_stat,
            weather_stats=wth_stat,
            train_pct=0.70,
        )
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        test_dataset = IdealPytorchDataset(
            home_ids,
            orchestrator,
            split="test",
            power_stats=pwr_stat,
            weather_stats=wth_stat,
            train_pct=0.70,
        )

        model = TemporalFusionTransformer(orchestrator.cardinalities, smoke_test=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=30
        )

        trainer = TemporalFusionTrainer(
            model, train_loader, val_loader, optimizer, scheduler, device=DEVICE
        )

        for epoch in range(EPOCHS):
            train_loss = trainer.train_epoch(accumulation_step)
            val_loss = trainer.validate(accumulation_step)
            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']}"
            )
            scheduler.step(val_loss)
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping triggered. Training stopped.")
                break

        trainer.plot_learning_curves()
        # visualize_tft_rolling_week(
        #     model,
        #     val_dataset,
        #     home_ids[split_idx],
        #     feature_names=STATIC_FEATURES,
        #     device=DEVICE,
        #     smoke_test=True,
        # )
        test_ids = [sample['homeid'] for sample in test_dataset.samples]

        for test_id in test_ids:
            visualize_density_heatmap(model, test_dataset, test_id, device=DEVICE)
    else:
        print("No data loaded. Check DATA_DIR path.")

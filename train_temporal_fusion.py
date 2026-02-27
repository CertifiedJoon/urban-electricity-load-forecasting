from src.Orchestrator.IdealDatasetOrchestrator import IdealDatasetOrchestrator
from src.Transformer.TemporalFusionTransformer import TemporalFusionTransformer
from src.PytorchDataset.IdealPytorchDataset import IdealPytorchDataset
from src.Trainer.TemporalFusionTrainer import TemporalFusionTrainer
from src.Trainer.EarlyStopper import EarlyStopping
from src.interpret import visualize_tft_rolling_week
from torch.utils.data import DataLoader
import torch
import random
import os

if __name__ == "__main__":
    # Settings
    DATA_DIR = "data"  # Update this path
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
    BATCH_SIZE = 8  # Good for 16GB VRAM
    EPOCHS = 2000
    LR = 5e-6
    STATIC_FEATURES = [
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

    # 1. Pipeline Setup
    orchestrator = IdealDatasetOrchestrator(DATA_DIR)
    early_stopping = EarlyStopping(patience=300, verbose=True, save_path="model.pth")

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
    split_idx = int(len(home_ids) * 0.8)
    train_ids = home_ids[:split_idx]
    val_ids = home_ids[split_idx:]

    print("1. Train + Interpret\n2. Interpret\n3. Smoke Test\nType 1 or 2 or 3:")
    choice = int(input())
    # choice = 1
    if choice == 1:
        val_dataset = IdealPytorchDataset(val_ids, orchestrator)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        train_dataset = IdealPytorchDataset(train_ids, orchestrator)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        # 2. Model Setup
        model = TemporalFusionTransformer(orchestrator.cardinalities)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        # ReduceLROnPlateau => reduce learning rate when loss plateus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20
        )

        # 3. Training & Eval
        trainer = TemporalFusionTrainer(
            model, train_loader, val_loader, optimizer, scheduler, device=DEVICE
        )

        for epoch in range(EPOCHS):
            train_loss = trainer.train_epoch()
            val_loss = trainer.validate()
            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )
            scheduler.step(val_loss)
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping triggered. Training stopped.")
                break

        # Plot learning curve
        trainer.plot_learning_curves()

        # Interpret results
        interpret_batch = next(iter(val_loader))
        visualize_tft_rolling_week(
            model,
            val_dataset,
            home_ids[split_idx],
            feature_names=STATIC_FEATURES,
            device=DEVICE,
        )
    elif choice == 2:
        val_dataset = IdealPytorchDataset(val_ids, orchestrator)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print("Input .pth path. current path is " + os.getcwd() + ":")
        model_path = input()
        model = TemporalFusionTransformer(orchestrator.cardinalities)
        model.load_state_dict(torch.load(model_path, map_location="cuda"))
        model.to("cuda")
        interpret_batch = next(iter(val_loader))
        visualize_tft_rolling_week(
            model,
            val_dataset,
            home_ids[split_idx],
            feature_names=STATIC_FEATURES,
            device=DEVICE,
        )
    elif choice == 3:
        print("RUNNING IN SMOKE TEST MODE (CPU)")
        # Overwrite config for speed
        BATCH_SIZE = 2
        EPOCHS = 1
        MAX_BATCHES_PER_EPOCH = 5

        # Just take the first 4 homes
        train_dataset = IdealPytorchDataset(train_ids, orchestrator)
        val_dataset = IdealPytorchDataset(val_ids, orchestrator)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 2. Model Setup
        model = TemporalFusionTransformer(orchestrator.cardinalities, smoke_test=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        # ReduceLROnPlateau => reduce learning rate when loss plateus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=30
        )

        # 3. Training & Eval
        trainer = TemporalFusionTrainer(
            model, train_loader, val_loader, optimizer, scheduler, device=DEVICE
        )

        for epoch in range(EPOCHS):
            train_loss = trainer.train_epoch()
            val_loss = trainer.validate()
            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )
            scheduler.step(val_loss)
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping triggered. Training stopped.")
                break

        # Plot learning curve
        trainer.plot_learning_curves()

        # Interpret results
        interpret_batch = next(iter(val_loader))
        visualize_tft_rolling_week(
            model,
            val_dataset,
            home_ids[split_idx],
            feature_names=STATIC_FEATURES,
            device=DEVICE,
        )
    else:
        print("No data loaded. Check DATA_DIR path.")

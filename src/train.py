from Orchestrator.IdealDatasetOrchestrator import IdealDatasetOrchestrator
from PytorchDataset.IdealPytorchDataset import IdealPytorchDataset
from Transformer.SocioTemporalTransformer import InterpretableSocioTransformer
from Trainer.IdealTrainer import IdealTrainer
from Trainer.EarlyStopper import EarlyStopping
from torch.utils.data import DataLoader
from interpret import visualize_rolling_week
import torch
import random
import os

if __name__ == "__main__":
    # Settings
    DATA_DIR = "../data" # Update this path
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
    BATCH_SIZE = 64 # Good for 16GB VRAM
    EPOCHS = 500
    
    # 1. Pipeline Setup
    orchestrator = IdealDatasetOrchestrator(DATA_DIR)
    early_stopping = EarlyStopping(patience=50, verbose=True, save_path="model.pth")
    
    # Select home IDs (In real usage, list available IDs from file)
    home_ids = [int(filename.split("_", 1)[0][4:]) for filename in filter(lambda x: x.endswith(".csv"), os.listdir("../data/household_sensors/"))]

    random.seed(42) # For reproducibility
    random.shuffle(home_ids)

    # 80/20 Split
    split_idx = int(len(home_ids) * 0.8)
    train_ids = home_ids[:split_idx]
    val_ids = home_ids[split_idx:]

    print("1. Train + Interpret\n2. Interpret \nType 1 or 2:")
    choice = int(input())

    val_dataset = IdealPytorchDataset(val_ids, orchestrator)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    # choice = 1
    
    if choice == 1 and len(train_dataset) > 0:
        train_dataset = IdealPytorchDataset(train_ids, orchestrator)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        # 2. Model Setup
        model = InterpretableSocioTransformer(orchestrator.cardinalities)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        # ReduceLROnPlateau => reduce learning rate when loss plateus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=30
        )
        
        # 3. Training & Eval
        trainer = IdealTrainer(model, train_loader, val_loader, optimizer, scheduler, device=DEVICE)
        
        for epoch in range(EPOCHS):
            train_loss = trainer.train_epoch(epoch)
            val_loss = trainer.validate()
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            scheduler.step(val_loss)
            early_stopping(val_loss, model)
            
            if early_stopping.early_stop:
                print("Early stopping triggered. Training stopped.")
                break
            
        # Plot learning curve
        trainer.plot_learning_curves()
        
        # Interpret results
        interpret_batch = next(iter(val_loader))
        visualize_rolling_week(model, val_dataset, home_ids[split_idx])
    elif choice == 2:
        print("Input .pth path. current path is " + os.getcwd() + ":")
        model_path = input()
        model = InterpretableSocioTransformer(orchestrator.cardinalities)
        model.load_state_dict(torch.load(model_path, map_location='cuda'))
        model.to('cuda')
        interpret_batch = next(iter(val_loader))
        visualize_rolling_week(model, val_dataset, home_ids[split_idx])
    else:
        print("No data loaded. Check DATA_DIR path.")
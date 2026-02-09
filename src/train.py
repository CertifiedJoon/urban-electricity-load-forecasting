from Orchestrator.IdealDatasetOrchestrator import IdealDatasetOrchestrator
from PytorchDataset.IdealPytorchDataset import IdealPytorchDataset
from Transformer.SocioTemporalTransformer import SocioTemporalTransformer
from Trainer.IdealTrainer import IdealTrainer
from Trainer.EarlyStopper import EarlyStopping
from torch.utils.data import DataLoader
import torch
import random
import os



if __name__ == "__main__":
    # Settings
    DATA_DIR = "../data" # Update this path
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
    BATCH_SIZE = 4 # Good for 16GB VRAM
    EPOCHS = 500
    
    # 1. Pipeline Setup
    orchestrator = IdealDatasetOrchestrator(DATA_DIR)
    early_stopping = EarlyStopping(patience=50, verbose=True, save_path="model.pth")
    
    # Select home IDs (In real usage, list available IDs from file)
    home_ids = [int(filename.split("_", 1)[0][4:]) for filename in filter(lambda x: x.endswith(".csv"), os.listdir("../data/household_sensors/"))]

    random.seed(42) # For reproducibility
    random.shuffle(home_ids)

    # 80/20 Split
    split_idx = int(len(home_ids) * 0.9)
    train_ids = home_ids[:split_idx]
    val_ids = home_ids[split_idx:]

    train_dataset = IdealPytorchDataset(train_ids, orchestrator)
    val_dataset = IdealPytorchDataset(val_ids, orchestrator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    if len(train_dataset) > 0:
        # 2. Model Setup
        model = SocioTemporalTransformer()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, 
        #                                                 steps_per_epoch=len(train_loader), 
        #                                                 epochs=EPOCHS)
        
        # ReduceLROnPlateau => reduce learning rate when loss plateus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
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
            
        trainer.plot_learning_curves()
    else:
        print("No data loaded. Check DATA_DIR path.")
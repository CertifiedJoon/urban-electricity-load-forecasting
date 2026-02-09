import numpy as np
import torch
import os

class EarlyStopping:
    """Stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, min_delta=0, save_path='checkpoint.pt', verbose=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                               Default: 0
            save_path (str): Path for the checkpoint to be saved to.
            verbose (bool): If True, prints a message for each validation improvement. 
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        # 1. First Run: Initialize best_loss
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        # 2. Improvement Check
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0 # Reset counter
        # 3. No Improvement: Increment counter
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss
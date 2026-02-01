from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import os

class IdealDataProcessor(ABC):
    """Abstract Base Class for all IDEAL data processing."""
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Standard helper to load a CSV from the directory."""
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        return pd.read_csv(path)

    @abstractmethod
    def process(self, **kwargs) -> pd.DataFrame:
        """Force every subclass to define its own processing logic."""
        pass
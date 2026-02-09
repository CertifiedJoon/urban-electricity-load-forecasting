from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import os

class IdealDataProcessor(ABC):
    """Abstract Base Class for all IDEAL data processing."""
    def __init__(self, data_path):
        self.data_path = data_path
    
    @abstractmethod
    def process(self, **kwargs): pass
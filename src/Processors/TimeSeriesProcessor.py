from Processors.IdealDataProcessor import IdealDataProcessor
import pandas as pd
import re
import os
import numpy as np

class LoadProcessor(IdealDataProcessor):
    """
    Locates and processes the specific electric-combined file for a given home.
    Naming Convention: home[id]_[room]_[sensor_id]_electric-mains_electric-combined.csv.gz
    """
    def find_file_for_home(self, home_id):
        # We need to find the file that matches the pattern:
        pattern = re.compile(fr"home{home_id}*")
        
        if not os.path.exists(self.data_path):
            return None

        for filename in os.listdir(self.data_path):
            if pattern.match(filename):
                return os.path.join(self.data_path, filename)
        return None

    def process(self, home_id):
        file_path = self.find_file_for_home(home_id)
        
        if file_path is None:
            return None

        df = pd.read_csv(file_path)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Log-scaling for stability
        df['value'] = np.log1p(df['value']) 
        return df

class WeatherProcessor(IdealDataProcessor):
    def process(self, feed_id: str) -> pd.DataFrame:
        # Weather data comes in 15-minute intervals [cite: 139]
        df = self.load_csv(f"weather_{feed_id}.csv")
        return self._resample_and_clean(df)

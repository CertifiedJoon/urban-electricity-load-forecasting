from src.Processors.IdealDataProcessor import IdealDataProcessor
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
        pattern = re.compile(rf"^home{home_id}(?:_|$)")

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

        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            print(f"Home {home_id}: Skipped (Empty sensor file: {file_path})")
            return None
        except pd.errors.ParserError as exc:
            print(f"Home {home_id}: Skipped (Unreadable sensor file: {file_path}; {exc})")
            return None

        required_columns = {"timestamp", "value"}
        if not required_columns.issubset(df.columns):
            print(
                f"Home {home_id}: Skipped (Missing required columns {required_columns} in {file_path})"
            )
            return None

        if df.empty:
            print(f"Home {home_id}: Skipped (No rows in sensor file: {file_path})")
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Log-scaling for stability
        df["value"] = np.log1p(df["value"])
        df["hour"] = df["timestamp"].dt.hour
        df["dayofweek"] = df["timestamp"].dt.weekday
        df["month"] = df["timestamp"].dt.month
        df.set_index("timestamp", inplace=True)
        return df


class WeatherProcessor(IdealDataProcessor):
    def __init__(self, data_path):
        super().__init__(data_path)
        # Initialize weather data
        weather_df = pd.read_csv(os.path.join(self.data_path, "combined_weather.csv"))
        weather_df["time"] = pd.to_datetime(weather_df["time"])
        weather_df = weather_df[["time", "locationid", "temperature", "conditions"]]
        weather_df["conditions"] = pd.factorize(weather_df["conditions"])[0]
        self.weather_df_by_location_id = dict()

        for location, group in weather_df.groupby("locationid"):
            group = group.set_index("time").sort_index()
            resampled_group = (
                group[["temperature", "conditions"]].resample("1min").ffill().bfill()
            )
            self.weather_df_by_location_id[location] = resampled_group

        # Initialize home metadata (home metadata is used to join weather and power usage data)
        home_meta_df = pd.read_csv(os.path.join(self.data_path, "home.csv"))
        self.home_meta_df = home_meta_df[["homeid", "location"]].set_index("homeid")

    def process(self, home_id, freq="1min") -> pd.DataFrame:
        location = self.home_meta_df.loc[home_id, "location"]
        if location in self.weather_df_by_location_id:
            return self.weather_df_by_location_id[location]
        raise ValueError(
            f"Location for given home_id - {str(home_id)} - not found in weather data"
        )

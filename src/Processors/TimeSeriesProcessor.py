from src.Processors import IdealDataProcessor
import pandas as pd


class TimeSeriesProcessor(IdealDataProcessor):
    def __init__(self, data_dir: str, freq: str = "1T"):
        super().__init__(data_dir)
        self.freq = freq

    def _resample_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Common internal logic for time-series cleaning."""
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df.resample(self.freq).mean().interpolate(limit=5).fillna(0)


class LoadProcessor(TimeSeriesProcessor):
    def process(self, sensor_id: str) -> pd.DataFrame:
        # datasheet specifies 'electric-combined' for total load [cite: 126]
        df = self.load_csv(f"sensor_{sensor_id}.csv")
        df = self._resample_and_clean(df)

        # Apply Log-Scaling as discussed for model stability
        df["value"] = np.log1p(df["value"])
        return df


class WeatherProcessor(TimeSeriesProcessor):
    def process(self, feed_id: str) -> pd.DataFrame:
        # Weather data comes in 15-minute intervals [cite: 139]
        df = self.load_csv(f"weather_{feed_id}.csv")
        return self._resample_and_clean(df)

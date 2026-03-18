from src.Processors.MetadataProcessor import MetadataProcessor
from src.Processors.TimeSeriesProcessor import LoadProcessor
from src.Processors.TimeSeriesProcessor import WeatherProcessor


class IdealDatasetOrchestrator:
    """Coordinating class that prepares paired (Static, Dynamic) samples."""

    def __init__(self, data_dir):
        self.meta_proc = MetadataProcessor(data_dir + "/metadata_and_surveys/metadata/")
        self.load_proc = LoadProcessor(data_dir + "/household_sensors/")
        self.weather_proc = WeatherProcessor(data_dir + "/weather/")
        self.cached_meta, self.cardinalities = self.meta_proc.process()

    def get_home_data(self, home_id):
        # Get static data
        static_row = self.cached_meta[self.cached_meta["homeid"] == home_id]
        # Get time_series data
        dynamic_df = self.load_proc.process(home_id)
        weather_df = self.weather_proc.process(home_id)
        dynamic_df = dynamic_df.join(weather_df, how="inner")

        return static_row.iloc[0], dynamic_df

from src.Processors.MetadataProcessor import MetadataProcessor
from src.Processors.TimeSeriesProcessor import LoadProcessor
from src.Processors.TimeSeriesProcessor import WeatherProcessor


class IdealDatasetOrchestrator:
    """Coordinating class that prepares paired (Static, Dynamic) samples."""

    def __init__(self, data_dir):
        self.meta_proc = MetadataProcessor(data_dir + "/metadata_and_surveys/metadata/")
        self.load_proc = LoadProcessor(data_dir + "/household_sensors/")
        self.cached_meta, self.cardinalities = self.meta_proc.process()

    def get_home_data(self, home_id):
        # 1. Get Static DNA
        if self.cached_meta is None:
            return None, None

        static_row = self.cached_meta[self.cached_meta["homeid"] == home_id]
        if static_row.empty:
            return None, None

        # 2. Get Dynamic Stream
        dynamic_df = self.load_proc.process(home_id)
        if dynamic_df is None:
            return None, None

        return static_row.iloc[0], dynamic_df

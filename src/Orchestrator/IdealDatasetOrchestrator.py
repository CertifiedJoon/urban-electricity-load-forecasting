from src.Processors.MetadataProcessor import MetadataProcessor
from src.Processors.TimeSeriesProcessor import LoadProcessor
from src.Processors.TimeSeriesProcessor import WeatherProcessor


class IdealDatasetOrchestrator:
    def __init__(self, data_dir: str):
        self.meta_proc = MetadataProcessor(data_dir)
        self.load_proc = LoadProcessor(data_dir)
        self.wthr_proc = WeatherProcessor(data_dir)

    def build_home_profile(self, home_id: str, sensor_id: str, feed_id: str):
        # Polymorphic execution
        static_data = self.meta_proc.process()
        home_static = static_data[static_data["homeid"] == home_id]

        load_data = self.load_proc.process(sensor_id)

        weather_data = self.wthr_proc.process(feed_id)

        full_dynamic = load_data.join(weather_data, how="inner")

        return home_static, full_dynamic

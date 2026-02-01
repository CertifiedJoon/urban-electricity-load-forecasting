from src.Processors import IdealDataProcessor
import pandas as pd


class MetadataProcessor(IdealDataProcessor):
    def process(self) -> pd.DataFrame:
        # Load tables identified in the datasheet
        home_df = self.load_csv("home.csv")
        person_df = self.load_csv("person.csv")
        app_df = self.load_csv("appliance.csv")

        # 1. Filter relevant home fields [cite: 38]
        home_cols = ["homeid", "residents", "income band", "hometype", "build era"]
        home_df = home_df[home_cols]

        # 2. Extract Primary Participant Info [cite: 78]
        primary = person_df[person_df["primaryparticipant"] == True]
        primary = primary[["homeid", "workingstatus", "ageband"]]

        # 3. Aggregate high-power appliance counts [cite: 102]
        key_apps = ["electricshower", "washingmachine", "dishwasher", "tumbledrier"]
        app_counts = app_df[app_df["appliancetype"].isin(key_apps)]
        app_summary = app_counts.pivot_table(
            index="homeid", columns="appliancetype", aggfunc="size", fill_value=0
        )

        # Polymorphic merge
        final_meta = home_df.merge(primary, on="homeid", how="left")
        final_meta = final_meta.merge(app_summary, on="homeid", how="left").fillna(0)
        return final_meta

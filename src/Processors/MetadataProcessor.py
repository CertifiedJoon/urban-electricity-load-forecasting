from Processors.IdealDataProcessor import IdealDataProcessor
import pandas as pd
import os

class MetadataProcessor(IdealDataProcessor):
    def process(self):
        # Load tables identified in the datasheet
        home_path = os.path.join(self.data_path, "home.csv")
        person_path = os.path.join(self.data_path, "person.csv")
        
        # Check if files exist
        if not os.path.exists(home_path):
            print(f"Error: Metadata file not found at {home_path}")
            return None

        home = pd.read_csv(home_path)
        person = pd.read_csv(person_path)
        
        # 1. Core Home Info
        # Selecting specific columns relevant to the model
        # Note: 'income band' usually has a space in the raw csv [cite: 38]
        home_cols = ['homeid', 'residents', 'income_band', 'hometype']
        meta = home[home_cols].copy()
        
        # 2. Primary Participant Info
        # We only want the person who is the 'primaryparticipant' [cite: 78]
        primary = person[person['primaryparticipant'] == True][['homeid', 'workingstatus']]
        meta = meta.merge(primary, on='homeid', how='left')

        # 3. Factorize Categoricals for Embedding Layers
        # Filling N/A values before conversion
        for col in ['income_band', 'hometype', 'workingstatus']:
            meta[col] = meta[col].fillna("Unknown")
            meta[col] = pd.factorize(meta[col])[0]
            
        return meta
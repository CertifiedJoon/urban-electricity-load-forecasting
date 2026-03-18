import pandas as pd
import os

def combine_weather_data(readings_path, metadata_path, output_path):
    """
    Combines raw IDEAL weather readings with feed metadata to create 
    a single wide-format CSV (Time x Location -> Features).
    """
    print("Loading weather files...")
    
    # 1. Load the CSVs based on your specification
    # parse_dates=['time'] ensures the UTC timestamp is read correctly
    readings = pd.read_csv(readings_path, parse_dates=['time'])
    metadata = pd.read_csv(metadata_path)

    # 2. Merge Metadata to Readings
    # We join on 'feedid' to attach 'weather_type', 'locationid', etc. to each reading
    print("Merging metadata...")
    merged = pd.merge(readings, metadata, on='feedid', how='left')

    # 3. Pivot to Wide Format
    # The goal is to have one row per (Time, Location) containing all weather metrics.
    # index: Unique identifiers for a row
    # columns: What we want to become the new column headers (weather_type)
    # values: The actual data points
    print("Pivoting to wide format (this may take a moment)...")
    
    weather_wide = merged.pivot_table(
        index=['time', 'locationid'], 
        columns='weather_type', 
        values='value',
        aggfunc='first'  # Logic: There should only be one value per type per time
    ).reset_index()

    # 4. Clean Up
    # The pivot might create NaN values if some feeds are missing at certain timestamps.
    # We forward-fill (ffill) then backward-fill (bfill) to handle small gaps.
    weather_wide.sort_values(by=['locationid', 'time'], inplace=True)
    weather_wide = weather_wide.groupby('locationid').apply(lambda x: x.ffill().bfill())
    
    # Reset index after group operation
    weather_wide.reset_index(drop=True, inplace=True)

    # 5. Save
    print(f"Saving combined weather data to {output_path}...")
    weather_wide.to_csv(output_path, index=False)
    print("Done. Sample output:")
    print(weather_wide.head())

# --- Usage Example ---
if __name__ == "__main__":
    # Update these paths to where your actual CSV files are located
    READINGS_FILE = "weatherreading.csv"  # The file with time, value, feedid
    METADATA_FILE = "metadata_and_surveys/metadata/weatherfeed.csv"     # The file with weather_type, locationid
    OUTPUT_FILE = "combined_weather.csv"
    
    # Create dummy files for demonstration if they don't exist
    if not os.path.exists(READINGS_FILE):
        print("Creating dummy data for demonstration...")
        # Dummy Metadata
        meta_data = {
            'feedid': [1, 2, 3, 4],
            'weather_type': ['temperature', 'humidity', 'temperature', 'humidity'],
            'locationid': [1, 1, 2, 2],
            'unit': ['C', '%', 'C', '%'],
            'source': ['Station A', 'Station A', 'Station B', 'Station B']
        }
        pd.DataFrame(meta_data).to_csv(METADATA_FILE, index=False)
        
        # Dummy Readings (15 min intervals as per datasheet)
        dates = pd.date_range('2018-01-01', periods=5, freq='15T')
        readings_data = []
        for date in dates:
            readings_data.append([1, date, 10.5]) # Loc 1 Temp
            readings_data.append([2, date, 55.0]) # Loc 1 Hum
            readings_data.append([3, date, 11.2]) # Loc 2 Temp
            readings_data.append([4, date, 60.0]) # Loc 2 Hum
            
        pd.DataFrame(readings_data, columns=['feedid', 'time', 'value']).to_csv(READINGS_FILE, index=False)

    combine_weather_data(READINGS_FILE, METADATA_FILE, OUTPUT_FILE)
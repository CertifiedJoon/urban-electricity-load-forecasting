import gzip
import pandas as pd
import os

done_files = set(filter(lambda x: x.endswith(".csv"), os.listdir(".")))
print(str(len(done_files)) + " already done")
for file in os.listdir("."):
  print("Check if done: " + file)
  search_done_file = file.removesuffix(".gz")
  if file.endswith(".gz") and search_done_file not in done_files:
    print("Opening...")
    with gzip.open(file) as f:
      df = pd.read_csv(f, header=None, names=['timestamp', 'value'])
      df['timestamp'] = pd.to_datetime(df['timestamp'])
      df.set_index('timestamp', inplace=True)
      df = df.resample('1Min').sum()
      df.to_csv(file.removesuffix(".gz"))
      print(df.head())
      print(file + " Done")
    

import kagglehub

# Download latest version
path = kagglehub.dataset_download("alonhaviv/the-maestro-dataset-v3-0-0")

print("Path to dataset files:", path)

import pandas as pd
import os

meta = pd.read_csv(f"{path}/maestro-v3.0.0/maestro-v3.0.0.csv")

# Faccio la somma di tutte le duration nelle righe
total_duration = meta["duration"].sum()
print("Total duration (seconds):", total_duration, "minutes:", total_duration / 60, "hours:", total_duration / 3600)
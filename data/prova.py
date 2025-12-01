import pandas as pd

db = pd.read_csv('musicnet_metadata.csv')

# Filtra i pezzi con ensemble = "Solo Piano"
piano_solo = db[db["ensemble"] == "Solo Piano"]

# Somma i secondi
total_seconds = piano_solo["seconds"].sum()

print("Totale secondi Piano Solo:", total_seconds)

total_hours = total_seconds / 3600

print("Totale ore Piano Solo:", total_hours)
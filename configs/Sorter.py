import pandas as pd
import csv

# Apro il file che rappresenterà il nostro mini_dataset
file_mini = open("musicnet_metadata_sp.csv", "w")

# Creo un iterable per il file originale
df = pd.read_csv("musicnet_metadata.csv", delimiter=",")

# Creo la prima riga del nuovo file
labels = ','.join(f'"{col}"' for col in df.columns)

# Scrivo sul file i campi
file_mini.write(labels)

# Ciclo per iterare dentro df
for index, row in df.iterrows():
    # Controllo se la riga corrisponde a Solo Piano
    if row["ensemble"] == "Solo Piano":
        # Se si vado a scrivere la riga nel mio file
        row_index = df.iloc[index]
        row_str = ','.join(f'"{str(v)}"' for v in row.values)
        file_mini.write(row_str + "\n")

# Chiudo il file su cui stavo scrivendo
file_mini.close()
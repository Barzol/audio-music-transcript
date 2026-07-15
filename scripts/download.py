import kagglehub
import pandas as pd
from pathlib import Path
import os

def main():
    print("Download MAESTRO da Kaggle...")

    path = kagglehub.dataset_download("alonhaviv/the-maestro-dataset-v3-0-0")
    print(f"Dataset created in: {path}")

    root = Path(path) / "maestro-v3.0.0"
    meta_path = root / "maestro-v3.0.0.csv"
    
    if not meta_path.exists(): 
        print(f"Error: file {meta_path} not found")
        return
    
    df = pd.read_csv(meta_path)
    
    df['full_midi_path'] = df['midi_filename'].apply(lambda x : str(root / x))
    df['full_audio_path'] = df['audio_filename'].apply(lambda x : str(root / x))
    
    print(f"Dataset loaded : {len(df)} tracks ready for training.")

    print(f"Path example : {df['full_midi_path'].iloc[0]}")

if __name__ == "__main__":
    main()
import kagglehub
import pandas as pd
import random
import os
from pathlib import Path
from collections import defaultdict

TEST_RATIO = 0.15
SEED       = 42


def stratified_split(df, test_ratio, seed):
    """
    Divide il DataFrame in train/test in modo stratificato per compositore.
    Garantisce almeno 1 traccia di test per ogni compositore presente.

    Il compositore viene estratto dal percorso MIDI:
      .../musicnet_midis/musicnet_midis/<Compositore>/<file>.mid
    """
    random.seed(seed)

    def get_composer(midi_path):
        parts = Path(midi_path).parts
        idx = [i for i, p in enumerate(parts) if p == "musicnet_midis"]
        return parts[idx[-1] + 1] if idx and idx[-1] + 1 < len(parts) else "Unknown"

    df = df.copy()
    df["composer"] = df["midi_path"].apply(get_composer)

    new_splits = []

    for composer, group in df.groupby("composer"):
        ids = group.index.tolist()
        random.shuffle(ids)

        n_test = max(1, round(len(ids) * test_ratio))
        test_ids  = set(ids[:n_test])
        train_ids = set(ids[n_test:])

        print(f"  {composer:12s}: {len(train_ids):3d} train, {len(test_ids):2d} test "
              f"(tot {len(ids)}, test {len(test_ids)/len(ids)*100:.0f}%)")

        for idx in ids:
            new_splits.append((idx, "test" if idx in test_ids else "train"))

    split_map = dict(new_splits)
    df["split"] = df.index.map(split_map)
    df.drop(columns=["composer"], inplace=True)
    return df


def main():
    print("Download MusicNet da Kaggle...")

    path = kagglehub.dataset_download("imsparsh/musicnet-dataset")
    print(f"Dataset scaricato in: {path}")

    root      = Path(path) / "musicnet" / "musicnet"
    meta_path = Path(path) / "musicnet_metadata.csv"
    midi_root = Path(path) / "musicnet_midis" / "musicnet_midis"

    meta = pd.read_csv(meta_path)

    print("Ricerca file WAV, label e MIDI...")
    records = []

    for _, row in meta.iterrows():
        track_id = str(row["id"])
        ensemble = row["ensemble"]

        if   (root / "train_data" / f"{track_id}.wav").exists():
            src_split = "train"
        elif (root / "test_data"  / f"{track_id}.wav").exists():
            src_split = "test"
        else:
            continue

        midi_files = list(midi_root.rglob(f"{track_id}*.mid*"))
        if not midi_files:
            print(f"  Warning: MIDI non trovato per track {track_id}, skip")
            continue

        records.append({
            "id":         track_id,
            "split":      src_split,
            "ensemble":   ensemble,
            "wav_path":   str(root / f"{src_split}_data"   / f"{track_id}.wav"),
            "label_path": str(root / f"{src_split}_labels" / f"{track_id}.csv"),
            "midi_path":  str(midi_files[0]),
        })

    df = pd.DataFrame(records)

    solo_piano = df[df["ensemble"] == "Solo Piano"].copy().reset_index(drop=True)
    print(f"\nTracce Solo Piano trovate: {len(solo_piano)}")

    print(f"\nSplit stratificato (test_ratio={TEST_RATIO}, seed={SEED}):")
    solo_piano = stratified_split(solo_piano, TEST_RATIO, SEED)

    n_train = (solo_piano["split"] == "train").sum()
    n_test  = (solo_piano["split"] == "test").sum()
    print(f"\n  → TOTALE: {n_train} train, {n_test} test "
          f"({n_test/(n_train+n_test)*100:.1f}% test)")

    os.makedirs("data", exist_ok=True)
    out_path = "data/solo_piano.csv"
    solo_piano.to_csv(out_path, index=False)
    print(f"\nSalvato {out_path} con {len(solo_piano)} tracce.")


if __name__ == "__main__":
    main()
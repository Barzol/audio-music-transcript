import kagglehub
import pandas as pd
import random
import os
from pathlib import Path
from collections import defaultdict
 
# ── Parametri dello split ─────────────────────────────────────────────────────
TEST_RATIO = 0.15   # 15% test
VAL_RATIO  = 0.15   # 15% val (calcolato sulle tracce train rimaste dopo il test)
SEED       = 42     # seed per riproducibilità
#
# Split risultante (su ~156 tracce):
#   test  ≈ 24 tracce  (15% del totale)
#   val   ≈ 20 tracce  (15% del restante 85%)
#   train ≈ 112 tracce (rimanenti)
 
 
def stratified_split(df, test_ratio, val_ratio, seed):
    """
    Divide il DataFrame in train/val/test in modo stratificato per compositore.
    Garantisce almeno 1 traccia per split per ogni compositore presente.
 
    Procedura:
      1. Per ogni compositore: separa il test (test_ratio sul totale)
      2. Dalle rimanenti: separa il val (val_ratio sulle rimanenti)
      3. Il resto diventa train
 
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
 
    split_assignments = {}
 
    for composer, group in df.groupby("composer"):
        ids = group.index.tolist()
        random.shuffle(ids)
 
        n_total = len(ids)
        n_test  = max(1, round(n_total * test_ratio))
        n_val   = max(1, round((n_total - n_test) * val_ratio))
 
        test_ids  = set(ids[:n_test])
        val_ids   = set(ids[n_test:n_test + n_val])
        train_ids = set(ids[n_test + n_val:])
 
        print(f"  {composer:12s}: {len(train_ids):3d} train, {len(val_ids):2d} val, "
              f"{len(test_ids):2d} test  (tot {n_total})")
 
        for idx in ids:
            if idx in test_ids:
                split_assignments[idx] = "test"
            elif idx in val_ids:
                split_assignments[idx] = "val"
            else:
                split_assignments[idx] = "train"
 
    df["split"] = df.index.map(split_assignments)
    df.drop(columns=["composer"], inplace=True)
    return df
 
 
def main():
    print("Download MusicNet da Kaggle...")
 
    # ── Download ──────────────────────────────────────────────────────────────
    path = kagglehub.dataset_download("imsparsh/musicnet-dataset")
    print(f"Dataset scaricato in: {path}")
 
    root      = Path(path) / "musicnet" / "musicnet"
    meta_path = Path(path) / "musicnet_metadata.csv"
    midi_root = Path(path) / "musicnet_midis" / "musicnet_midis"
 
    meta = pd.read_csv(meta_path)
 
    # ── Costruzione lista tracce ──────────────────────────────────────────────
    print("Ricerca file WAV, label e MIDI...")
    records = []
 
    for _, row in meta.iterrows():
        track_id = str(row["id"])
        ensemble = row["ensemble"]
 
        # Determina la cartella sorgente (train o test di MusicNet)
        # NB: questo split originale verrà sovrascritto dopo
        if   (root / "train_data" / f"{track_id}.wav").exists():
            src_split = "train"
        elif (root / "test_data"  / f"{track_id}.wav").exists():
            src_split = "test"
        else:
            continue    # file non trovato, salta
 
        midi_files = list(midi_root.rglob(f"{track_id}*.mid*"))
        if not midi_files:
            print(f"  Warning: MIDI non trovato per track {track_id}, skip")
            continue
 
        records.append({
            "id":         track_id,
            "split":      src_split,            # verrà sovrascritto
            "ensemble":   ensemble,
            "wav_path":   str(root / f"{src_split}_data"   / f"{track_id}.wav"),
            "label_path": str(root / f"{src_split}_labels" / f"{track_id}.csv"),
            "midi_path":  str(midi_files[0]),
        })
 
    df = pd.DataFrame(records)
 
    # ── Filtra Solo Piano ─────────────────────────────────────────────────────
    solo_piano = df[df["ensemble"] == "Solo Piano"].copy().reset_index(drop=True)
    print(f"\nTracce Solo Piano trovate: {len(solo_piano)}")
 
    # ── Split stratificato per compositore ────────────────────────────────────
    print(f"\nSplit stratificato (test={TEST_RATIO}, val={VAL_RATIO}, seed={SEED}):")
    solo_piano = stratified_split(solo_piano, TEST_RATIO, VAL_RATIO, SEED)
 
    n_train = (solo_piano["split"] == "train").sum()
    n_val   = (solo_piano["split"] == "val").sum()
    n_test  = (solo_piano["split"] == "test").sum()
    n_tot   = n_train + n_val + n_test
    print(f"\n  → TOTALE: {n_train} train, {n_val} val, {n_test} test "
          f"({n_train/n_tot*100:.0f}% / {n_val/n_tot*100:.0f}% / {n_test/n_tot*100:.0f}%)")
 
    # ── Salvataggio CSV ───────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    out_path = "data/solo_piano.csv"
    solo_piano.to_csv(out_path, index=False)
    print(f"\nSalvato {out_path} con {len(solo_piano)} tracce.")
 
 
if __name__ == "__main__":
    main()
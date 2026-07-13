"""
create_maestro_subset.py

Crea un subset di MAESTRO stratificato per compositore,
mantenendo le proporzioni per ogni split originale.

Il dataset MAESTRO completo (~121 GB) resta dov'e' (es. cache kagglehub);
questo script legge solo il suo maestro-v3.0.0.json e produce un CSV
leggero con lo stesso schema di maestro-v3.0.0.csv (canonical_composer,
canonical_title, split, year, midi_filename, audio_filename, duration),
cosi' da poter essere usato direttamente come --dataset.csv_file in
configs/config.yaml, lasciando dataset.data_dir puntato alla cache
originale (midi_filename/audio_filename sono path relativi ad essa).

Output: data/maestro_subset.csv

Uso:
    python scripts/create_maestro_subset.py \
        --maestro_json /path/to/maestro-v3.0.0.json \
        --n_train 115 \
        --n_val 20 \
        --n_test 25 \
        --seed 42 \
        --output data/maestro_subset.csv
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


def safe_print(text: str) -> None:
    """Print robusto a caratteri non rappresentabili nella codepage
    della console Windows (es. nomi di compositori con accenti)."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc))

CSV_COLUMNS = [
    "canonical_composer",
    "canonical_title",
    "split",
    "year",
    "midi_filename",
    "audio_filename",
    "duration",
]

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "maestro_subset.csv"


def load_maestro(json_path: str) -> list[dict]:
    """Carica il JSON di Maestro e restituisce una lista di record."""
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "data" in data:
            return data["data"]

        # maestro-v3.0.0.json e' column-oriented (come pandas
        # to_json(orient="columns")): {colonna: {row_idx: valore}}
        if set(CSV_COLUMNS).issubset(data.keys()):
            row_ids = data[CSV_COLUMNS[0]].keys()
            return [
                {col: data[col][rid] for col in CSV_COLUMNS}
                for rid in row_ids
            ]

        # fallback: dict indicizzato {row_idx: record}
        return list(data.values())

    raise ValueError(f"Formato JSON non riconosciuto: {type(data)}")


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    """
    Campiona n record da una lista con stratificazione per compositore.
    Se n >= len(records), restituisce tutti i record (shuffled).
    """
    if n >= len(records):
        result = records.copy()
        random.seed(seed)
        random.shuffle(result)
        return result

    # Raggruppa per compositore
    by_composer: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        composer = rec.get("canonical_composer", "Unknown")
        by_composer[composer].append(rec)

    composers = list(by_composer.keys())
    n_composers = len(composers)

    # Quota base per compositore
    base_quota = n // n_composers
    remainder = n % n_composers

    rng = random.Random(seed)

    # Ordine casuale dei compositori per distribuire il remainder
    composer_order = composers.copy()
    rng.shuffle(composer_order)

    sampled: list[dict] = []
    for i, composer in enumerate(composer_order):
        pool = by_composer[composer].copy()
        rng.shuffle(pool)
        quota = base_quota + (1 if i < remainder else 0)
        # Non possiamo prendere più di quanti ne abbiamo
        quota = min(quota, len(pool))
        sampled.extend(pool[:quota])

    # Se il campionamento per compositore ha dato meno di n
    # (alcune quote ridotte per pool piccoli), integrare dal resto
    if len(sampled) < n:
        sampled_keys = {
            (r["canonical_composer"], r.get("midi_filename", "")) for r in sampled
        }
        leftover = [
            r for r in records
            if (r["canonical_composer"], r.get("midi_filename", "")) not in sampled_keys
        ]
        rng.shuffle(leftover)
        sampled.extend(leftover[: n - len(sampled)])

    return sampled


def create_subset(
    json_path: str,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    output_path: str,
) -> None:
    records = load_maestro(json_path)
    print(f"Totale tracce Maestro: {len(records)}")

    # Separa per split originale
    splits: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        splits[rec["split"]].append(rec)

    print(f"  train: {len(splits['train'])} | "
          f"validation: {len(splits['validation'])} | "
          f"test: {len(splits['test'])}")

    # Campiona con stratificazione
    targets = {"train": n_train, "validation": n_val, "test": n_test}
    subset_records: list[dict] = []

    for split_name, n_target in targets.items():
        pool = splits[split_name]
        sampled = stratified_sample(pool, n_target, seed=seed + hash(split_name) % 1000)
        print(
            f"  Campionato {len(sampled)}/{n_target} da {split_name} "
            f"({len(set(r['canonical_composer'] for r in sampled))} compositori)"
        )
        subset_records.extend(sampled)

    # Serializza in CSV con lo stesso schema di maestro-v3.0.0.csv,
    # cosi' MaestroDataset puo' leggerlo senza modifiche.
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(subset_records)[CSV_COLUMNS]
    df.to_csv(out_path, index=False)

    print(f"\nSubset salvato in: {out_path}")
    print(f"Totale tracce nel subset: {len(subset_records)}")

    # Stampa distribuzione compositori per split
    print("\nDistribuzione compositori nel subset:")
    for split_name in ["train", "validation", "test"]:
        split_recs = [r for r in subset_records if r["split"] == split_name]
        composers = sorted(set(r["canonical_composer"] for r in split_recs))
        print(f"  {split_name} ({len(split_recs)} tracce, "
              f"{len(composers)} compositori):")
        for c in composers:
            n = sum(1 for r in split_recs if r["canonical_composer"] == c)
            safe_print(f"    {c}: {n}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crea un subset stratificato di MAESTRO"
    )
    parser.add_argument(
        "--maestro_json",
        type=str,
        required=True,
        help="Path al file maestro-v3.0.0.json (o v2)",
    )
    parser.add_argument("--n_train", type=int, default=115, help="Tracce di train (default: 115)")
    parser.add_argument("--n_val",   type=int, default=20,  help="Tracce di validation (default: 20)")
    parser.add_argument("--n_test",  type=int, default=25,  help="Tracce di test (default: 25)")
    parser.add_argument("--seed",    type=int, default=42,  help="Random seed (default: 42)")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Path output CSV (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_subset(
        json_path=args.maestro_json,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
        output_path=args.output,
    )
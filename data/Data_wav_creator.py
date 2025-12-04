# crea_subset_musicnet_tar_o_dir.py
import tarfile
from pathlib import Path
import pandas as pd
import io
import os

# CONFIG - modifica se serve
# Percorsi che ho utilizzato per importare gli archivi
# Possono essere incapsulati dentro un file di config per renderlo portabile

# Percorso del file CSV_SP
metadata_csv = "Dataset_sp/musicnet_metadata_sp.csv"
# Percorso dell'archivio originale
source_path = Path("musicnet.tar.gz")
# Archivio destinazione   
dest_tar = "musicnet_sp.tar.gz"
# Questo va a creare le cartelle splittando tra testing and training
include_splits = ("train", "test")

# Funzione per caricare tutti gli id dei file Solo Piano dal file CSV creato 
def load_ids(csv_path):
    # Creo l'iterabile
    df = pd.read_csv(csv_path, encoding="latin1")
    # Controllo per verificare che effettivamente l'id sia dentro il CSV
    if "id" not in df.columns:
        # Lancia un'eccezzione
        raise ValueError("CSV senza colonna 'id'")
    # Altrimenti ritorna un set con l'id dentro 
    return set(str(x).strip() for x in df["id"].tolist())

# Funzione che Serve quando non hai un file sul disco da copiare, 
# ma già hai i suoi contenuti in memoria
def add_bytes_to_tar(tar_obj, arcname, data, src_member=None):
    # Crea un TarInfo per descrivere il file con arcname nome/path che il file avrà dentro il tar 
    # (es. "musicnet/train_data/2104.wav").
    # TarInfo() contiene tutte le informazioni sul file: nome, permessi, proprietario, data modifica, dimensione…
    info = tarfile.TarInfo(arcname)

    # Il tar deve sapere quanto “pesa” il file prima di scriverlo.
    info.size = len(data)
    # copia alcuni metadati se disponibili
    # src_member contiene i metadati dell'altro archivio poichè già esistente
    if src_member is not None:
        # Itero sugli attributi per ottenere i metadati
        for attr in ("mtime", "mode", "uid", "gid", "uname", "gname"):
            val = getattr(src_member, attr, None)
            # Se val non è un valore nullo cerca di settare gli attributi
            if val is not None:
                # Prova a settare altrimenti passa avanti
                try:
                    setattr(info, attr, val)
                except Exception:
                    pass
    # Scrive un nuovo file nel nuovo archivio dove passa le informazioni del file 
    # E la dimensione del file            
    tar_obj.addfile(info, io.BytesIO(data))

# Funzione che va a ricercare i file per ids se presenti li copia nel nuovo archivio
# Prende in input il percorso sorgente, destinazione e il set contenente gli id su cui cercare
def process_from_dir(src_dir: Path, ids:set, dst):
    # Variabile che verrà incrementata per dare un feedback finale per capire se sono stati aggiunti
    # Tutti i file aspettati nel nostro caso dovranno essere 155  
    added = 0
    # Vado a creare i percorsi per i dati e le labels
    for split in include_splits:
        data_dir = src_dir / f"{split}_data"
        labels_dir = src_dir / f"{split}_labels"

        # Controllo se il percorso esiste
        if data_dir.exists():
            # Va a scorrere tutti i file dentro il percorso
            for f in data_dir.iterdir():
                # Vado a prendere solo file che terminano con .wav
                if not f.is_file() or f.suffix.lower() != ".wav":
                    continue
                # Controlla se l'id corrisponde con quelli che stiamo cercando
                # Se si vado ad aggiungere il file dentro il nuovo archivio
                if f.stem in ids:
                    # Percorso completo
                    arcname = f"musicnet/{split}_data/{f.name}"
                    # Aggiungo il file alla directory
                    add_bytes_to_tar(dst, arcname, f.read_bytes())
                    # Incremento il contatore
                    added += 1
                    print("Aggiunto (dir):", arcname)

        # Stessa cosa fatta per i file wav viene fatta per i file CSV ovvero le labels
        if labels_dir.exists():
            for f in labels_dir.iterdir():
                if not f.is_file() or f.suffix.lower() != ".csv":
                    continue
                if f.stem in ids:
                    arcname = f"musicnet/{split}_labels/{f.name}"
                    add_bytes_to_tar(dst, arcname, f.read_bytes())
                    added += 1
                    print("Aggiunto (dir):", arcname)
    # Ritorna il contatore                 
    return added

# Questa funzione va a prenere i file dall'archivio seguendo le regole e lo copia nel nuovo archivio
def process_from_tar(src_tar_path: Path, ids:set, dst):
    added = 0
    # Apre l'archivio
    with tarfile.open(src_tar_path, "r:gz") as src:
        # Scorre tutti i membri
        for member in src.getmembers():
            # Se non è un normale file lo salta
            if not member.isreg():      # salta directory, link, ecc.
                continue
            # Normalizzazione del nome
            name = member.name.lstrip("./")   # normalizza
            lower = name.lower()
            # Vogliamo solo file sotto musicnet/<split>_data o <split>_labels
            if not any(f"/{split}_data/" in lower or f"/{split}_labels/" in lower for split in include_splits):
                continue
            # Estrae l'ID dal nome del file per fare il confronto
            base = os.path.basename(name)
            stem, ext = os.path.splitext(base)
            id_ = stem.split("_")[0]   # robustezza se ci sono underscore
            # Controlla che sia dentro il set degli ids
            if id_ in ids:
                # Se lo trova lo estrae
                fobj = src.extractfile(member)
                if fobj is None:
                    print("Impossibile leggere member:", name)
                    continue
                # Lo copia su data
                data = fobj.read()
                fobj.close()
                # Manteniamo lo stesso nome interno (normalizzato) dentro l'archivio di destinazione
                arcname = name if name.startswith("musicnet/") else f"musicnet/{name}"
                # Lo va a scrivere dentro il nuovo archivio
                add_bytes_to_tar(dst, arcname, data, src_member=member)
                # Incremente il totale
                added += 1
                print("Aggiunto (tar):", arcname)
    # Ritorna il totale
    return added

# Funzione main che chiama tutte le funzioni precedenti
# Flusso funzione: 
# 1.Carica gli id da CSV
# 2.Apre la sorgente "Archivio Musicnet.tar.gz"
# 3.Copia i file selezionati attraverso gli id
# 4.Stampa il risultato finale
def main():
    # Carico gli id
    ids = load_ids(metadata_csv)
    print(f"ID caricati: {len(ids)}")

    # Controllo se esiste la sorgente
    if not source_path.exists():
        raise FileNotFoundError(f"Sorgente non trovata: {source_path}")

    added = 0
    # Apro l'archivio di destinazione
    with tarfile.open(dest_tar, "w:gz") as dst:
        # La sorgente è una cartella
        if source_path.is_dir():
            print("Sorgente è una cartella; procedo dalla directory.")
            added = process_from_dir(source_path, ids, dst)
        # La sorgente è un archivio    
        elif source_path.is_file() and source_path.suffixes[-2:] == [".tar", ".gz"]:
            print("Sorgente è un tar.gz; procedo dall'archivio.")
            added = process_from_tar(source_path, ids, dst)
        # La sorgente non è valida "ERRORE"
        else:
            raise ValueError("Sorgente non riconosciuta: deve essere una cartella 'musicnet' o un tar.gz")

    # Stampo resoconto finale
    print(f"Fatto: {dest_tar} creato ({added} file aggiunti).")

# Chiama il main se il file è eseguito come script, come nel mio caso
if __name__ == "__main__":
    main()

# Importo la libreria per gestire gli archivi
import tarfile, csv, os, re, io
import pandas as pd

# Creo un nuovo archivio dove andrò a salvare i file che mi interessano
with tarfile.open("musicnet_midis_sp.tar.gz", "w:gz"):
    pass

# Creo l'iterabile del file csv
df = pd.read_csv("musicnet_metadata_sp.csv", delimiter=",", encoding="latin1")

# Sorgenti per i due archivi
source_tar = "musicnet_midis.tar.gz"
dest_tar   = "musicnet_midis_sp.tar.gz"

# Prepara l'insieme di id come stringhe "pulite"
ids = set(str(x).strip() for x in df["id"].tolist())

# Con la libreria apro l'archivio e lo soprannomino tar
with tarfile.open(source_tar, "r:gz") as src, tarfile.open(dest_tar, "w:gz") as dst:

    # Vado a prendere tutti i file contenuti in src    
    for member in src.getmembers():
            # Vado a controllare che non siano cartelle per sicurezza e portabilità
            if not member.isreg():
                continue

            # Vado ad estrarre il nome del file da cui otterrò il codice
            filename = os.path.basename(member.name)

            # Prendo il numero facendo lo split fino a trovare / e _
            filename = filename.split("/")[-1]
            number = filename.split("_")[0]  # Fallback

             # Se il numero è nella lista degli id -> copialo
            if number in ids:
                fileobj = src.extractfile(member)   # File-like stream
                if fileobj is None:
                    continue  # In rari casi può essere None (es. member non è un file leggibile)7

                data = fileobj.read()
                # fileobj.close()

                # Crea un nuovo TarInfo con size corretto
                info = tarfile.TarInfo(name=member.name)
                info.size = len(data)
                info.mtime = getattr(member, "mtime", None)
                info.mode = getattr(member, "mode", None)
                # Copia metadata opzionali se presenti
                try:
                    info.uname = member.uname
                    info.gname = member.gname
                    info.uid = member.uid
                    info.gid = member.gid
                except Exception:
                    pass

                # Aggiungo il file all'archivio di destinazione
                dst.addfile(member, io.BytesIO(data))

                print(f"Copiato: {member.name}")   

    # Chiudo il file-like per liberare risorse
    fileobj.close() 

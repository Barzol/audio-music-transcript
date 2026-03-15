import argparse
import sys
import os

# Aggiungiamo le cartelle al path di Python per poter importare i moduli
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importiamo le funzioni principali dai vari file che abbiamo creato
from download_and_filter import main as download_data
from build_dataset import main as build_data
from train import train
from evaluate import evaluate

def main():
    # Creiamo il parser per i comandi da terminale
    parser = argparse.ArgumentParser(description="Progetto ML: Automatic Music Transcription (Solo Piano)")
    
    # Definiamo gli argomenti possibili
    parser.add_argument('--download', action='store_true', help="Step 1: Scarica il dataset e crea solo_piano.csv")
    parser.add_argument('--build', action='store_true', help="Step 2: Estrae i file del piano in data/raw/")
    parser.add_argument('--train', action='store_true', help="Step 3: Avvia l'addestramento della CRNN")
    parser.add_argument('--evaluate', action='store_true', help="Step 4: Valuta le metriche sul test set")

    args = parser.parse_args()

    # Eseguiamo l'azione richiesta
    if args.download:
        print("--- AVVIO DOWNLOAD E FILTRAGGIO ---")
        download_data()
    elif args.build:
        print("--- AVVIO COSTRUZIONE DATASET LOCALE ---")
        build_data()
    elif args.train:
        print("--- AVVIO ADDESTRAMENTO ---")
        train()
    elif args.evaluate:
        print("--- AVVIO VALUTAZIONE ---")
        evaluate()
    else:
        # Se non viene passato nessun argomento, mostra l'aiuto
        parser.print_help()

if __name__ == "__main__":
    main()
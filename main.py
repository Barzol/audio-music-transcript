import argparse
import sys
import os

# Aggiungiamo le cartelle al path di Python per poter importare i moduli
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importiamo le funzioni principali dai vari file che abbiamo creato
from train import train
from evaluate import evaluate

def main():
    # Creiamo il parser per i comandi da terminale
    parser = argparse.ArgumentParser(description="ML Project : Automatic Music Transcription")
    
    # Definiamo gli argomenti possibili
    parser.add_argument('--download', action='store_true', help="Step 1: Download dataset and create solo_piano.csv")
    parser.add_argument('--build', action='store_true', help="Step 2: Extract piano files in data/raw/")
    parser.add_argument('--train', action='store_true', help="Step 3: Start Training")
    parser.add_argument('--evaluate', action='store_true', help="Step 4: Evaluation")
    parser.add_argument('--plot', action='store_true', help="Step 5: Plot results")


    args = parser.parse_args()

    # Eseguiamo l'azione richiesta
    if args.download:
        print("--- DOWNLOAD AND FILTER ---")
        download_data()
    elif args.build:
        print("--- BUILDING LOCAL DATASET ---")
        build_data()
    elif args.train:
        print("--- TRAINING ---")
        train()
    elif args.evaluate:
        print("--- EVALUATION ---")
        evaluate()
    else:
        # Se non viene passato nessun argomento, mostra l'aiuto
        parser.print_help()

if __name__ == "__main__":
    main()
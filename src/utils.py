
import torch
import random
import numpy as np
import os
import yaml

# Qui si può mettere la funzione che controlla se si sta
# usando Cuda o Cpu

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Oppure si può mettere una funzione set_seed() che fissa
# a random i seed di torch, numpy e random, questo è utile
# così ogni volta che si allena il modello non si ottengono
# risultati diversi.

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Pytorch ha torch.utils.tensorboard che tiene traccia dei
# log delle varie loss durante il training. I file di log 
# finiscono automaticamente in runs/ 

def save_checkpoint(state, filename="my_checkpoint.pth.tar", dir_path="checkpoints"):
    filepath = os.path.join(dir_path, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved at {filepath}")

def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    print("Checkpoint loaded successfully.")
    return checkpoint # Ritorna il dizionario intero nel caso servano info extra (es. epoca)

def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


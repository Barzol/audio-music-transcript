# Qui dentro viene messa la classe Dataset
# Deve contenere i metodi
# __init__
# __len__
# __getitem__
# Per fare Data Augmentation è meglio definirle Qui
# Il vantaggio è che possiamo importare dataset senza preoccuparci
# di come vengono letti i file CSV, le immagini o i file audio

import kagglehub

# Download latest version
path = kagglehub.dataset_download("imsparsh/musicnet-dataset")

print("Path to dataset files:", path)


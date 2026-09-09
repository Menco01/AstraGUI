import gdown
import os

# Configurazione
FOLDER_ID = '16EHJpWeifbXTK4lWKYiXEpccBhgG21zu'
OUTPUT_DIR = 'data/Mayo_Dataset'

print("Inizio il download del Mayo's Dataset da Google Drive...")

# Crea la cartella data se non esiste
os.makedirs('data', exist_ok=True)

# Download della cartella
gdown.download_folder(id=FOLDER_ID, output=OUTPUT_DIR, quiet=False, use_cookies=False)

print(f"Download completato! I dati si trovano in: {OUTPUT_DIR}")

# ASTRA GUI: Ricostruzione e Denoising di Immagini Mediche

Applicazione desktop sviluppata in Python e PyQt5 per la ricostruzione tomografica (Computed Tomography) e l'analisi di immagini mediche e scientifiche. Il motore computazionale è basato sull'**ASTRA Toolbox** per l'esecuzione della Filtered Back Projection (FBP), supportato da algoritmi classici e reti neurali convoluzionali in PyTorch per il denoising avanzato (Learned Post-Processing).

---

## 🚀 Caratteristiche Principali

* **Ricostruzione CT:** Filtered Back Projection (FBP) con ASTRA Toolbox, personalizzabile per numero di angoli, densità di campionamento e frequenza di taglio.
* **Denoising Tradizionale:** Filtri Mediano, Bilaterale, Total Variation e Non-Local Means implementati tramite SciPy, OpenCV e scikit-image.
* **Learned Post-Processing (LPP):** Denoising avanzato basato su architetture U-Net (2 e 3 livelli con Batch Normalization) implementate in PyTorch. Modelli pre-addestrati caricabili direttamente nella GUI.
* **Generazione Dati:** Creazione di dati sintetici tramite fantoccio di Shepp-Logan personalizzabile.
* **Metriche di Valutazione:** Analisi quantitativa tramite PSNR, SSIM e MSE. Analisi visiva tramite estrazione di plot di riga, calcolo di media/deviazione standard e visualizzazione di istogrammi.

---

## 📁 Struttura della Repository

```text
astra-gui/
│
├── data/                       # Destinazione del Mayo's Dataset scaricato
│   └── .gitkeep
│
├── models/                     # Modelli pre-addestrati e script di training
│   ├── Model Training/         # Script per la generazione dataset e cicli di training
│   │   ├── fbp_module.py
│   │   ├── medical_model_trainer_unet_2.py
│   │   ├── medical_model_trainer_unet_3.py
│   │   └── metrics_module.py
│   └── Ready Models/           # Archiviazione dei pesi dei modelli pre-addestrati (.pth)
│
├── Analysis/
│   └── analysis_menu.py        # Dialoghi GUI per l'analisi quantitativa e visiva
├── DataGeneration/
│   └── data_generation_menu.py # Interfaccia per generazione fantoccio di Shepp-Logan
├── Denoise/
│   └── denoise_menu.py         # Interfaccia per filtri di denoising classici
├── Reconstruction/
│   ├── fbp_lpp_menu_unet_2.py   # Pipeline FBP + U-Net 2 livelli
│   ├── fbp_lpp_menu_unet_3.py   # Pipeline FBP + U-Net 3 livelli
│   └── reconstruction_menu.py # Configurazione ed esecuzione FBP standard
│
├── denoise_module.py           # Algoritmi di filtraggio classico
├── fbp_module.py               # Generazione sinogramma e ricostruzione FBP
├── main_gui.py                 # Entry point dell'applicazione PyQt5
├── metrics_module.py           # Calcolo metriche di qualità (PSNR, SSIM, MSE)
│
├── .gitignore                  # Esclusione cache, modelli pesanti e dataset
├── environment.yml             # File di configurazione ambiente Conda
├── setup_env.bat               # Script di setup automatico per Windows
├── setup_env.sh                # Script di setup automatico per Linux/Mac
├── download_mayo.py            # Script Python per il download del Mayo's Dataset da GDrive
└── README.md (o README.txt)
```

---

## ⚙️ Installazione e Setup

Il progetto utilizza **Conda** per la gestione delle dipendenze, fondamentale per installare correttamente l'ASTRA Toolbox.

### 1. Clona la repository
```bash
git clone https://github.com/Menco01/AstraGUI.git
cd AstraGUI
```

### 2. Configura l'ambiente
Puoi utilizzare gli script automatici presenti nella cartella principale del progetto:

* **Windows:**
  Esegui il file batch facendo doppio clic su `setup_env.bat` oppure esegui da terminale:
  ```cmd
  setup_env.bat
  ```

* **Linux / macOS:**
  Da terminale, assegna i permessi ed esegui lo script:
  ```bash
  chmod +x setup_env.sh
  ./setup_env.sh
  ```

* **Manualmente (Alternativa):**
  Se preferisci configurare l'ambiente a mano senza script:
  ```bash
  conda env create -f environment.yml
  conda activate astraFBP
  ```

### 3. Download del Dataset
Il *Mayo's Dataset* necessario per l'addestramento e il test risiede su Google Drive. Assicurati che l'ambiente Conda sia attivo ed esegui lo script `download_mayo.py` direttamente dalla root:

```bash
conda activate astraFBP
python download_mayo.py
```

---

## 🖥 Avvio dell'Applicazione

Assicurati che l'ambiente Conda sia attivo e lancia l'interfaccia grafica eseguendo il file principale dalla root del progetto:

```bash
conda activate astraFBP
python main_gui.py
```

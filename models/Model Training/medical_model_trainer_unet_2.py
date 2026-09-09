import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt

# IMPORT DEI MODULI ESTERNI

from fbp_module import image_FBP
from metrics_module import calculate_PSNR, calculate_SSIM, calculate_MSE


# CONFIGURAZIONE HYPERPARAMETERS

CONFIG = {
    # Percorsi
    "training_path": "Mayo_s Dataset Halved/train",
    "test_path": "Mayo_s Dataset Halved/test",
    
    # Hyperparameters Training
    "image_size": 512,
    "batch_size": 4,
    "epochs": 15,
    "learning_rate": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Hyperparameters FBP (ASTRA)
    "fbp_angles": 30,
    "fbp_det_spacing": 1.0,
    "fbp_cutoff": 1.0
}

folder_name = f"training_{CONFIG['fbp_angles']}_{CONFIG['fbp_det_spacing']}_{CONFIG['fbp_cutoff']}_unet2"

os.makedirs(folder_name, exist_ok=True)

CONFIG["model_save_path"] = os.path.join(folder_name, "denoising_unet_model.pth")
CONFIG["loss_plot_path"] = os.path.join(folder_name, "training_loss_plot.png")

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Cartella creata: {folder_name}")

# DEFINIZIONE DEL DATASET

class MedicalImageDataset(Dataset):
    def __init__(self, folder_path, config, num_samples=None):
        """
        Carica le immagini, esplorando anche le sottocartelle, applica la FBP al volo
        e restituisce la coppia (Input FBP, Target GT).
        """
        self.config = config

        # 1. Crea il pattern di ricerca ricorsivo
        search_pattern = os.path.join(folder_path, "**", "*.*")
        all_files = glob.glob(search_pattern, recursive=True)

        # 2. Filtra solo i file che sono effettivamente immagini
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        self.image_paths = [f for f in all_files if f.lower().endswith(valid_extensions)]

        if num_samples is not None:
            self.image_paths = self.image_paths[:num_samples]
            
        self.count = len(self.image_paths)
        if self.count == 0:
            # Stampiamo un messaggio evidente ma non solleviamo eccezioni qui 
            # per permettere alla logica di train/test di gestire il fallback
            print(f"\n[ERRORE DATASET] Nessuna immagine valida trovata in: {folder_path}")

        if len(self.image_paths) == 0:
            print(f"ATTENZIONE: Nessuna immagine trovata in {folder_path} o nelle sue sottocartelle")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Carica la Ground Truth in scala di grigi
        gt_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gt_img is None:
            raise ValueError(f"Impossibile leggere l'immagine: {img_path}")

        # Ridimensiona a 512x512 come richiesto
        gt_img = cv2.resize(gt_img, (self.config["image_size"], self.config["image_size"]))

        # Normalizza tra [0, 1]
        gt_img = gt_img.astype(np.float32) / 255.0

        # Applica FBP tramite il modulo esterno
        # Assumiamo che restituisca (sinogram, fbp_res) come visto in FBPDialog
        _, fbp_img = image_FBP(
            gt_img,
            self.config["fbp_angles"],
            self.config["fbp_det_spacing"],
            self.config["fbp_cutoff"]
        )

        # Converti in tensori PyTorch (aggiungendo la dimensione del canale [C, H, W])
        gt_tensor = torch.from_numpy(gt_img).unsqueeze(0).float()
        fbp_tensor = torch.from_numpy(fbp_img).unsqueeze(0).float()

        return fbp_tensor, gt_tensor
    
    # MODELLO DI DENOISING (U-NET)

class DenoiseUNet(nn.Module):
    def __init__(self):
        super(DenoiseUNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(64, 128)

        # Decoder
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64) # 128 perché concateniamo enc2 (64) + upconv2 (64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)  # 64 perché concateniamo enc1 (32) + upconv1 (32)

        # Output layer
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        # Bottleneck
        b = self.bottleneck(self.pool2(e2))

        # Decoder con Skip Connections
        d2 = self.upconv2(b)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return torch.sigmoid(out) # Ritorna valori tra 0 e 1


# DEFINIZIONE LOSS

class SSIMLoss(nn.Module):
    """Loss basata sullo Structural Similarity Index (SSIM)"""
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size

    def forward(self, img1, img2):
        # Calcola lo SSIM (usando una semplificazione basata su pooling)
        mu1 = F.avg_pool2d(img1, self.window_size, stride=1, padding=self.window_size//2)
        mu2 = F.avg_pool2d(img2, self.window_size, stride=1, padding=self.window_size//2)
        
        sigma1_sq = F.avg_pool2d(img1 * img1, self.window_size, stride=1, padding=self.window_size//2) - mu1.pow(2)
        sigma2_sq = F.avg_pool2d(img2 * img2, self.window_size, stride=1, padding=self.window_size//2) - mu2.pow(2)
        sigma12 = F.avg_pool2d(img1 * img2, self.window_size, stride=1, padding=self.window_size//2) - mu1 * mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1.pow(2) + mu2.pow(2) + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # La loss è (1 - SSIM) perché vogliamo minimizzarla
        return 1 - ssim_map.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.2):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        
        # Pesi per bilanciare le loss
        self.alpha = alpha   # Peso MSE
        self.beta = beta     # Peso L1
        self.gamma = gamma   # Peso SSIM

    def forward(self, prediction, target):
        loss_mse = self.mse(prediction, target)
        loss_l1 = self.l1(prediction, target)
        loss_ssim = self.ssim(prediction, target)
        
        return (self.alpha * loss_mse) + (self.beta * loss_l1) + (self.gamma * loss_ssim)

# FUNZIONI DI TRAINING E TEST

def train(config):
    print(f"--- Inizio Training su dispositivo: {config['device']} ---")

    # Dataloader
    train_dataset = MedicalImageDataset(config["training_path"], config)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    if len(train_dataset) == 0:
        print("--- Training Annullato: Dataset di addestramento vuoto. ---")
        return None  # Restituiamo None per indicare che non c'è un modello addestrato

    print(f"--- Inizio Training su {len(train_dataset)} immagini ---")

    # Inizializzazione Modello, Loss, Optimizer
    model = DenoiseUNet().to(config["device"])
    criterion = CombinedLoss(alpha=1.0, beta=0.5, gamma=0.5).to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    epoch_losses = []

    model.train()
    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        
        # Avvolgiamo il dataloader con tqdm
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config['epochs']}]", unit="batch")
        
        for fbp_inputs, gt_targets in progress_bar:
            fbp_inputs = fbp_inputs.to(config["device"])
            gt_targets = gt_targets.to(config["device"])
            
            # Forward pass
            outputs = model(fbp_inputs)
            loss = criterion(outputs, gt_targets)
            
            # Backward pass e ottimizzazione
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            epoch_loss += current_loss
            
            # Aggiorna la barra con la loss corrente
            progress_bar.set_postfix(loss=f"{current_loss:.6f}")
            
        avg_loss = epoch_loss / len(train_loader)

        epoch_losses.append(avg_loss)
        # La riga sotto stamperà la media a fine epoca senza sovrapporsi alla barra
        print(f" -> Media Loss Epoca {epoch+1}: {avg_loss:.6f}")

    # PLOTTING
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, config["epochs"] + 1), epoch_losses, marker='o', color='b', label='MSE Loss')
    plt.title("Andamento della Loss durante il Training")
    plt.xlabel("Epoca")
    plt.ylabel("Loss (MSE)")
    plt.grid(True)
    plt.legend()
    
    # Salva il grafico
    plt.savefig(config["loss_plot_path"])
    print(f"Grafico della loss salvato in: {config['loss_plot_path']}")
    plt.close() # Chiude la figura per liberare memoria

    # Salvataggio
    torch.save(model.state_dict(), config["model_save_path"])
    print(f"Training completato. Modello salvato in: {config['model_save_path']}\n")
    return model

def test(model, config):
    print("--- Inizio Fase di Test (Salvataggio Plot in 'test_results') ---")
    
    # 1. Crea la cartella per i risultati se non esiste
    output_dir = os.path.join(folder_name, "test_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataloader per il Test (limitato a 10 campioni)
    test_dataset = MedicalImageDataset(config["test_path"], config, num_samples=10)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if len(test_dataset) == 0:
        print("--- Test Annullato: Dataset di test vuoto. ---")
        return None

    model.eval()
    metrics = {"MSE": [], "PSNR": [], "SSIM": []}

    with torch.no_grad():
        for i, (fbp_input, gt_target) in enumerate(test_loader):
            fbp_input_dev = fbp_input.to(config["device"])

            # Inferenza
            output_tensor = model(fbp_input_dev)

            # Conversione in numpy per visualizzazione e metriche
            fbp_img = fbp_input.cpu().squeeze().numpy()
            gt_img = gt_target.cpu().squeeze().numpy()
            out_img = output_tensor.cpu().squeeze().numpy()

            # Calcolo Metriche
            mse_val = calculate_MSE(gt_img, out_img)
            psnr_val = calculate_PSNR(gt_img, out_img)
            ssim_val = calculate_SSIM(gt_img, out_img)

            metrics["MSE"].append(mse_val)
            metrics["PSNR"].append(psnr_val)
            metrics["SSIM"].append(ssim_val)

            # --- GENERAZIONE PLOT ---
            plt.figure(figsize=(15, 5))
            
            # Subplot 1: Ground Truth
            plt.subplot(1, 3, 1)
            plt.imshow(gt_img, cmap='gray')
            plt.title("Ground Truth (GT)")
            plt.axis('off')

            # Subplot 2: Input FBP (Sporca)
            plt.subplot(1, 3, 2)
            plt.imshow(fbp_img, cmap='gray')
            plt.title(f"Input FBP ({config['fbp_angles']} angles)")
            plt.axis('off')

            # Subplot 3: Output Modello (Denoised)
            plt.subplot(1, 3, 3)
            plt.imshow(out_img, cmap='gray')
            plt.title("Denoised (U-Net)")
            plt.axis('off')

            # Aggiunta metriche sotto il grafico
            footer_text = f"Metriche (GT vs Denoised):\nMSE: {mse_val:.6f} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}"
            plt.figtext(0.5, 0.02, footer_text, ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

            # Salvataggio
            save_path = os.path.join(output_dir, f"result_{i+1}.png")
            plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Lascia spazio per il testo in basso
            plt.savefig(save_path)
            plt.close() # Libera memoria

            print(f"Risultato {i+1} salvato in: {save_path}")

    # Stampa medie finali
    if len(metrics["MSE"]) > 0:
        print("\n--- Metriche Medie sul Test Set ---")
        print(f"Mean MSE:  {np.mean(metrics['MSE']):.6f}")
        print(f"Mean PSNR: {np.mean(metrics['PSNR']):.2f} dB")
        print(f"Mean SSIM: {np.mean(metrics['SSIM']):.4f}")

if __name__ == "__main__":
    # Assicurati che le cartelle esistano per evitare crash immediati
    os.makedirs(CONFIG["training_path"], exist_ok=True)
    os.makedirs(CONFIG["test_path"], exist_ok=True)

    # Avvia l'addestramento
    trained_model = train(CONFIG)

    # Avvia il test con il modello appena addestrato
    test(trained_model, CONFIG)
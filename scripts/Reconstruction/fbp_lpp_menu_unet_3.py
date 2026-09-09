import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QFileDialog, QSpinBox, QDoubleSpinBox, QLabel, QComboBox, QMessageBox, QWhatsThis)
from PyQt5.QtCore import Qt, QEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import moduli esterni
from fbp_module import image_FBP
from metrics_module import calculate_PSNR, calculate_SSIM, calculate_MSE

# Definizione Architettura
class DenoiseUNet(nn.Module):
    def __init__(self):
        super(DenoiseUNet, self).__init__()

        # Encoder (Più profondo)
        self.enc1 = self.conv_block(1, 64)   # Partiamo da 64 filtri
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(128, 256) 
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256) # 256(up) + 256(enc3) = 512 in ingresso

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128) # 128(up) + 128(enc2) = 256 in ingresso

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)  # 64(up) + 64(enc1) = 128 in ingresso

        # Output layer
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), # <-- AGGIUNTA FONDAMENTALE
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), # <-- AGGIUNTA FONDAMENTALE
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        b = self.bottleneck(self.pool3(e3))
        
        d3 = self.upconv3(b)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return torch.sigmoid(out)
    
class FBPLPPDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("FBP + LPP Reconstruction")
        self.resize(1000, 800)
        
        # Testo di aiuto
        self.help_text_fbp_lpp_unet3_window = (
            "<h3>Guida alla FBP + LPP Reconstruction (U-Net 3)</h3>"
            "<p>In questa sezione è possibile:</p>"
            "<ul>"
            "<li><b>Carica Modello:</b> Cliccando sul bottone è possibile selezionare un modello correttamente addestrato e compatibile alla U-Net utilizzata [U-Net 3] (per la lista dei modelli e per il training dei modelli, andare sulla cartella 'Models').</li>"
            "<li><b>Carica Immagine Singola:</b> Cliccando sul bottone è possibile selezionare l'immagine su cui svolgere la FBP + LPP.</li>"
            "<li><b>Carica Cartella:</b> Cliccando sul bottone è possibile selezionare la cartella sulle quali immagini la FBP + LPP (Di default verrà aperta la prima immagine). </li>"
            "<li><b>Configurazione Parametri per la FBP:</b> Utilizza i campi testuali per configurare i tuoi parametri per la FBP.</li>"
            "<li><b>Clicca Plot:</b> Cliccando sul plot o su una delle immagini nel plot, è possibile aprire una Scheda Immagine con l'immagine/plot selezionato.</li>"
            "<li><b>Navigazione Immagini:</b> Tramite il selettore e i bottoni ◀ | ▶ è possibile selezionare l'immagine della cartella caricata sulla quale svolgere la FBP + LPP.</li>"
            "</ul>"
        )

        self.model = None
        self.image_list = []
        self.current_idx = -1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # --- SEZIONE CARICAMENTO ---
        top_layout = QHBoxLayout()
        self.btn_load_model = QPushButton("Carica Modello (.pth)")
        self.btn_load_model.clicked.connect(self.load_model)
        
        self.btn_load_img = QPushButton("Carica Immagine Singola")
        self.btn_load_img.clicked.connect(lambda: self.load_data(single=True))
        
        self.btn_load_dir = QPushButton("Carica Cartella")
        self.btn_load_dir.clicked.connect(lambda: self.load_data(single=False))

        top_layout.addWidget(self.btn_load_model)
        top_layout.addWidget(self.btn_load_img)
        top_layout.addWidget(self.btn_load_dir)
        layout.addLayout(top_layout)

        self.lbl_info = QLabel("Carica modello e dati per iniziare")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_info)

        # --- SEZIONE PARAMETRI FBP ---
        fbp_params_layout = QHBoxLayout()
        
        # Configurazione Angoli (Intero)
        fbp_params_layout.addWidget(QLabel("Angoli:"))
        self.spin_angles = QSpinBox()
        self.spin_angles.setRange(10, 720)
        self.spin_angles.setValue(60)
        self.spin_angles.setToolTip("Numero di proiezioni angolari")
        self.spin_angles.valueChanged.connect(self.update_display) # Aggiorna al cambio
        fbp_params_layout.addWidget(self.spin_angles)

        # Configurazione Densità (Float)
        fbp_params_layout.addWidget(QLabel("Densità:"))
        self.spin_density = QDoubleSpinBox()
        self.spin_density.setRange(0.1, 5.0)
        self.spin_density.setSingleStep(0.1)
        self.spin_density.setValue(1.0)
        self.spin_density.setToolTip("Densità di campionamento dei detector")
        self.spin_density.valueChanged.connect(self.update_display)
        fbp_params_layout.addWidget(self.spin_density)

        # Configurazione Cutoff (Float)
        fbp_params_layout.addWidget(QLabel("Cutoff:"))
        self.spin_cutoff = QDoubleSpinBox()
        self.spin_cutoff.setRange(0.01, 1.0)
        self.spin_cutoff.setSingleStep(0.05)
        self.spin_cutoff.setValue(1.0)
        self.spin_cutoff.setToolTip("Frequenza di taglio del filtro passa-basso")
        self.spin_cutoff.valueChanged.connect(self.update_display)
        fbp_params_layout.addWidget(self.spin_cutoff)

        layout.addLayout(fbp_params_layout)

        # --- SEZIONE PLOT ---
        self.figure = Figure(figsize=(15, 8))
        self.canvas = FigureCanvas(self.figure)
        # Connessione dell'evento click
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        layout.addWidget(self.canvas)

        # --- SEZIONE NAVIGAZIONE ---
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_prev.setEnabled(False)

        self.combo_images = QComboBox()
        self.combo_images.currentIndexChanged.connect(self.select_from_combo)
        self.combo_images.setEnabled(False)

        self.btn_next = QPushButton("▶")
        self.btn_next.clicked.connect(self.next_image)
        self.btn_next.setEnabled(False)

        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.combo_images, 1) # Stretch 1
        nav_layout.addWidget(self.btn_next)
        layout.addLayout(nav_layout)

    def event(self, event):
        # Intercetta il momento in cui l'utente clicca il bottone "?"
        if event.type() == QEvent.EnterWhatsThisMode and self.isActiveWindow():
            # Questo ripristina all'istante il cursore a freccia normale.
            QWhatsThis.leaveWhatsThisMode()
            self.show_help_fbp_lpp_unet3_window()
            return True # Restituisce True per bloccare il cambio del cursore di default
        
        # Lascia che gli altri eventi vengano gestiti normalmente
        return super().event(event)

    def show_help_fbp_lpp_unet3_window(self):
        help_box = QMessageBox(self)
        help_box.setWindowTitle("Help")
        help_box.setTextFormat(Qt.RichText) # Permette di usare tag HTML come <b>, <p>, <ul>
        help_box.setText(self.help_text_fbp_lpp_unet3_window)
        help_box.setIcon(QMessageBox.Information)
        help_box.exec_()

    def on_plot_click(self, event):
        if event.inaxes is None:
            return

        # CASO 1: L'asse contiene immagini (imshow)
        if len(event.inaxes.images) > 0:
            img = event.inaxes.images[0].get_array()
            title = event.inaxes.get_title() or "Extracted Image"
            self.main_window.open_image_window(np.array(img), title)

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleziona Modello", "", "PyTorch Model (*.pth)")
        if path:
            try:
                self.model = DenoiseUNet().to(self.device)
                self.model.load_state_dict(torch.load(path, map_location=self.device))
                self.model.eval()
                self.lbl_info.setText(f"Modello caricato: {os.path.basename(path)}")
                self.update_display()
            except Exception as e:
                QMessageBox.critical(self, "Errore", f"Errore caricamento modello: Selezionare modello con U-Net corretta")

    def load_data(self, single=True):
        if single:
            path, _ = QFileDialog.getOpenFileName(self, "Seleziona Immagine", "", "Images (*.png *.jpg *.tif)")
            if path:
                self.image_list = [path]
                self.current_idx = 0
                self.toggle_nav(False)
        else:
            dir_path = QFileDialog.getExistingDirectory(self, "Seleziona Cartella")
            if dir_path:
                exts = (".png", ".jpg", ".tif", ".jpeg")
                self.image_list = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(exts)])
                if self.image_list:
                    self.current_idx = 0
                    self.combo_images.clear()
                    self.combo_images.addItems([os.path.basename(f) for f in self.image_list])
                    self.toggle_nav(True)
                else:
                    QMessageBox.warning(self, "Vuoto", "Nessuna immagine trovata nella cartella.")
        
        self.update_display()

    def toggle_nav(self, enabled):
        self.btn_prev.setEnabled(enabled)
        self.btn_next.setEnabled(enabled)
        self.combo_images.setEnabled(enabled)

    def update_display(self):
        if not self.model or self.current_idx == -1:
            return

        img_path = self.image_list[self.current_idx]
        gt_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gt_img is None: return
        
        # Forza la dimensione a 512x512 per renderla compatibile con la U-Net
        gt_img = cv2.resize(gt_img, (512, 512))
        
        # Normalizzazione
        gt_img = gt_img.astype(np.float32) / 255.0
        
        # --- LETTURA PARAMETRI DINAMICI ---
        n_angles = self.spin_angles.value()
        density = self.spin_density.value()
        cutoff = self.spin_cutoff.value()

        # 1. Calcolo FBP con parametri configurati
        # image_FBP(image, num_angles, sampling_density, cutoff_frequency)
        _, fbp_img = image_FBP(
            gt_img, 
            num_angles=n_angles, 
            sampling_density=density, 
            cutoff_frequency=cutoff
        )

        # 2. Denoise con Modello
        input_tensor = torch.from_numpy(fbp_img).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
            denoised_img = output_tensor.cpu().squeeze().numpy()

        # 3. Calcolo Metriche
        mse = calculate_MSE(gt_img, denoised_img)
        psnr = calculate_PSNR(gt_img, denoised_img)
        ssim = calculate_SSIM(gt_img, denoised_img)
        
        # 4. Plotting (Logica integrata stile PlotWindow)
        self.figure.clear()
        
        metrics_text = f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | MSE: {mse:.6f}"
        images = {
            "Ground Truth": gt_img, 
            f"FBP ({n_angles} ang)": fbp_img, 
            "FBP + LPP": denoised_img
        }
        
        # Creazione automatica dei subplot
        n_imgs = len(images)
        for i, (title, img_data) in enumerate(images.items()):
            ax = self.figure.add_subplot(1, n_imgs, i + 1)
            ax.imshow(img_data, cmap='gray')
            ax.set_title(title, fontsize=10)
            ax.axis('off')

        # Footer con le metriche
        self.figure.suptitle(metrics_text, y=0.05, fontsize=11, color='darkblue', fontweight='bold')
        
        # Ottimizzazione spazio (massimizza dimensioni immagini)
        self.figure.tight_layout(rect=[0, 0.12, 1, 0.98], w_pad=0.5)
        
        self.canvas.draw()

    def next_image(self):
        if self.current_idx < len(self.image_list) - 1:
            self.current_idx += 1
            self.combo_images.setCurrentIndex(self.current_idx)
            self.update_display()

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.combo_images.setCurrentIndex(self.current_idx)
            self.update_display()

    def select_from_combo(self, index):
        if index != -1:
            self.current_idx = index
            self.update_display()
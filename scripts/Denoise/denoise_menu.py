from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox, QComboBox
from denoise_module import (image_median_filter_denoise, image_bilateral_filter_denoise, 
                            image_total_variation_denoise, image_non_local_means_denoise)
from metrics_module import calculate_PSNR, calculate_SSIM, calculate_MSE

class DenoiseDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Image Denoise")
        self.setModal(False)
        layout = QVBoxLayout(self)
        
        if main_window.last_fbp_result is None:
            layout.addWidget(QLabel("ERRORE: Esegui prima una FBP. L'algoritmo richiede l'ultima FBP elaborata."))
            btn = QPushButton("Chiudi"); btn.clicked.connect(self.close)
            layout.addWidget(btn)
            return

        layout.addWidget(QLabel("Input: Ultima ricostruzione FBP salvata in memoria"))
        
        self.combo = QComboBox()
        self.combo.addItems(["Median Filter", "Bilateral Filter", "Total Variation", "Non-Local Means"])
        layout.addWidget(self.combo)
        
        btn_box = QHBoxLayout()
        btn_run = QPushButton("Denoise")
        btn_run.clicked.connect(self.run_denoise)
        btn_annulla = QPushButton("Annulla")
        btn_annulla.clicked.connect(self.close)
        btn_box.addWidget(btn_run); btn_box.addWidget(btn_annulla)
        layout.addLayout(btn_box)

    def run_denoise(self):
        gt = self.main_window.last_fbp_gt
        fbp_input = self.main_window.last_fbp_result
        algo = self.combo.currentText()
        
        if algo == "Median Filter": res = image_median_filter_denoise(fbp_input)
        elif algo == "Bilateral Filter": res = image_bilateral_filter_denoise(fbp_input)
        elif algo == "Total Variation": res = image_total_variation_denoise(fbp_input)
        else: res = image_non_local_means_denoise(fbp_input)
        
        psnr = calculate_PSNR(gt, res)
        ssim = calculate_SSIM(gt, res)
        mse = calculate_MSE(gt, res)
        
        metrics = f"Metriche Denoise -> PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | MSE: {mse:.6f}"
        images = {"FBP GT": gt, "FBP (Noisy)": fbp_input, f"Denoised ({algo})": res}
        
        from main_gui import PlotWindow
        PlotWindow(self.main_window, images_dict=images, info=metrics, title="Risultato Denoise").show()
        self.close()
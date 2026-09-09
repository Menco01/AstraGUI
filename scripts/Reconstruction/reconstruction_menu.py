from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QSpinBox, QLabel, QMessageBox, QWhatsThis
from PyQt5.QtCore import Qt, QEvent
from fbp_module import image_FBP
from metrics_module import calculate_PSNR, calculate_SSIM, calculate_MSE

class FBPDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Filter Back Projection")
        self.setModal(False)
        
        # Testo di aiuto
        self.help_text_reconstruction_window = (
            "<h3>Guida alla Filter Back Projection</h3>"
            "<p>In questa sezione è possibile:</p>"
            "<ul>"
            "<li><b>Slot Immagine Target:</b> Cliccando sul bottone e poi su una Scheda Immagine (ImageWindow), è possibile definire su quale immagine fare la FBP.</li>"
            "<li><b>Configurazione Parametri per la FBP:</b> Utilizza i campi testuali per configurare i tuoi parametri per la FBP.</li>"
            "<li><b>Reconstruct:</b> Svolgi la FBP sull'immagine scelta (o sulla sezione dell'immagine se è presente un Rettangolo). Verrà aperta una Scheda Plot contenente la Ground Truth, il Sinogramma e la FBP </li>"
            "</ul>"
        )

        # Sostituiamo target_img con target_window
        self.target_window = None 
        
        layout = QVBoxLayout(self)
        
        self.btn_slot = QPushButton("Immagine Target: Clicca qui, poi clicca un'ImageWindow")
        self.btn_slot.clicked.connect(self.prepare_slot)
        layout.addWidget(self.btn_slot)
        
        box_a = QHBoxLayout()
        box_a.addWidget(QLabel("Angoli di proiezione:"))
        self.spin_a = QSpinBox(); self.spin_a.setRange(10, 2000); self.spin_a.setValue(180)
        box_a.addWidget(self.spin_a)
        
        box_d = QHBoxLayout()
        box_d.addWidget(QLabel("Densità campionamento:"))
        self.spin_d = QSpinBox(); self.spin_d.setValue(1)
        box_d.addWidget(self.spin_d)
        
        box_f = QHBoxLayout()
        box_f.addWidget(QLabel("Freq. di taglio:"))
        self.spin_f = QSpinBox(); self.spin_f.setValue(1)
        box_f.addWidget(self.spin_f)
        
        btn_box = QHBoxLayout()
        btn_run = QPushButton("Reconstruct")
        btn_run.clicked.connect(self.run_fbp)
        btn_annulla = QPushButton("Annulla")
        btn_annulla.clicked.connect(self.close)
        btn_box.addWidget(btn_run); btn_box.addWidget(btn_annulla)
        
        for b in [box_a, box_d, box_f, btn_box]: layout.addLayout(b)

    def event(self, event):
        # Intercetta il momento in cui l'utente clicca il bottone "?"
        if event.type() == QEvent.EnterWhatsThisMode and self.isActiveWindow():
            # Questo ripristina all'istante il cursore a freccia normale.
            QWhatsThis.leaveWhatsThisMode()
            self.show_help_reconstruction_window()
            return True # Restituisce True per bloccare il cambio del cursore di default
        
        # Lascia che gli altri eventi vengano gestiti normalmente
        return super().event(event)

    def show_help_reconstruction_window(self):
        help_box = QMessageBox(self)
        help_box.setWindowTitle("Help")
        help_box.setTextFormat(Qt.RichText) # Permette di usare tag HTML come <b>, <p>, <ul>
        help_box.setText(self.help_text_reconstruction_window)
        help_box.setIcon(QMessageBox.Information)
        help_box.exec_()

    def prepare_slot(self):

        # Forza lo spegnimento dei bottoni nella UI principale
        if hasattr(self.main_window, 'disable_all_tools'):
            self.main_window.disable_all_tools()

        self.btn_slot.setText("Immagine Target: Clicca la finestra desiderata ora...")
        
        def callback(img, title, win_ref):
            self.target_window = win_ref
            self.btn_slot.setText(f"Target: {title}")
                
        self.main_window.image_selection_callback = callback

    def run_fbp(self):
        if self.target_window is None:
            QMessageBox.warning(self, "Errore", "Riempi lo slot cliccando prima su un'ImageWindow.")
            return

        if self.target_window.line_coords is not None:
            QMessageBox.warning(self, "Operazione non consentita", 
                                "La Filtered Back Projection non può essere eseguita se è presente una linea nell'immagine.\n"
                                "Rimuovi la linea o usa un rettangolo ROI.")
            return
    
        # RECUPERA LA ROI O L'INTERA IMMAGINE DINAMICAMENTE
        # Se c'è un rettangolo, get_active_image() restituirà solo quello
        target_img = self.target_window.get_active_image()

        if target_img is None or target_img.shape[0] < 2 or target_img.shape[1] < 2:
            QMessageBox.warning(self, "Errore ROI", 
                                "L'area selezionata è troppo piccola o non valida.\n"
                                "Riprova a disegnare il rettangolo o deselezionalo.")
            return
        try:
            # Esecuzione FBP
            sinogram, fbp_res = image_FBP(target_img, 
                                          self.spin_a.value(), 
                                          self.spin_d.value(), 
                                          self.spin_f.value())
        
            self.main_window.last_fbp_gt = target_img
            self.main_window.last_fbp_result = fbp_res
            
            psnr = calculate_PSNR(target_img, fbp_res)
            ssim = calculate_SSIM(target_img, fbp_res)
            mse = calculate_MSE(target_img, fbp_res)
            
            metrics = f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | MSE: {mse:.6f}"
            images = {"Ground Truth": target_img, "Sinogramma": sinogram, "FBP Result": fbp_res}
            
            from main_gui import PlotWindow
            PlotWindow(self.main_window, images_dict=images, info=metrics, title="Risultato FBP").show()
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Errore ASTRA", f"Errore durante la ricostruzione: {str(e)}")


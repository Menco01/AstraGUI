import numpy as np
import cv2
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox, QLabel, QWhatsThis
from PyQt5.QtCore import Qt, QEvent

def get_target(main_window):
    """
    Funzione di utilità per recuperare l'immagine o la ROI 
    dall'ultima ImageWindow che ha ricevuto focus.
    """
    if not main_window.last_focused_window:
        QMessageBox.warning(main_window, "Errore", "Seleziona prima una ImageWindow (cliccaci sopra).")
        return None
    return main_window.last_focused_window.get_active_image()


class ImageDifferenceDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Image Difference")
        self.setModal(False)
        self.img1, self.img2 = None, None
        
        # Testo di aiuto
        self.help_text_image_difference_window = (
            "<h3>Guida alla Image Difference</h3>"
            "<p>In questa sezione è possibile:</p>"
            "<ul>"
            "<li><b>Slot Immagine Target 1:</b> Cliccando sul bottone e poi su una Scheda Immagine (ImageWindow), è possibile definire la prima immagine su cui fare la differenza.</li>"
            "<li><b>Slot Immagine Target 2:</b> Cliccando sul bottone e poi su una Scheda Immagine (ImageWindow), è possibile definire la seconda immagine su cui fare la differenza.</li>"
            "<li><b>Esegui Differenza:</b> Svolgi la differenza tra le immagini scelte. Verrà aperta una Scheda Immagine contenente il risultato </li>"
            "</ul>"
        )

        layout = QVBoxLayout(self)
        self.btn1 = QPushButton("Slot 1: Seleziona ImageWindow")
        self.btn1.clicked.connect(lambda: self.prepare_slot(1))
        layout.addWidget(self.btn1)
        
        self.btn2 = QPushButton("Slot 2: Seleziona ImageWindow")
        self.btn2.clicked.connect(lambda: self.prepare_slot(2))
        layout.addWidget(self.btn2)
        
        btn_diff = QPushButton("Esegui Differenza")
        btn_diff.clicked.connect(self.compute_diff)
        layout.addWidget(btn_diff)

    def event(self, event):
        # Intercetta il momento in cui l'utente clicca il bottone "?"
        if event.type() == QEvent.EnterWhatsThisMode and self.isActiveWindow():
            # Questo ripristina all'istante il cursore a freccia normale.
            QWhatsThis.leaveWhatsThisMode()
            self.show_help_image_difference_window()
            return True # Restituisce True per bloccare il cambio del cursore di default
        
        # Lascia che gli altri eventi vengano gestiti normalmente
        return super().event(event)

    def show_help_image_difference_window(self):
        help_box = QMessageBox(self)
        help_box.setWindowTitle("Help")
        help_box.setTextFormat(Qt.RichText) # Permette di usare tag HTML come <b>, <p>, <ul>
        help_box.setText(self.help_text_image_difference_window)
        help_box.setIcon(QMessageBox.Information)
        help_box.exec_()

    def prepare_slot(self, slot_id):
        # DISATTIVA I TOOL prima di entrare in selezione
        self.main_window.disable_all_tools()
        
        self.main_window.image_selection_callback = lambda img, title, win_ref: self.set_image(title, slot_id, win_ref)

    def set_image(self, title, slot_id, win_ref):
        # Estraiamo l'immagine completa 2D ignorando eventuali ritagli/linee
        full_image = win_ref.image_data 
        
        if slot_id == 1:
            self.img1 = full_image
            self.btn1.setText(f"Slot 1: {title}")
        else:
            self.img2 = full_image
            self.btn2.setText(f"Slot 2: {title}")

    def compute_diff(self):
        if self.img1 is None or self.img2 is None:
            QMessageBox.warning(self, "Errore", "Seleziona entrambe le immagini.")
            return
        if self.img1.shape != self.img2.shape:
            QMessageBox.warning(self, "Errore", "Dimensioni diverse!")
            return
        
        diff = np.abs(self.img1.astype(float) - self.img2.astype(float))
        from main_gui import PlotWindow
        PlotWindow(self.main_window, images_dict={"Differenza (Abs)": diff}, title="Result").show()

class PlotCommand(QDialog):
    """Produce il plot del profilo di riga se è presente una Linea."""
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Plot Linea")
        layout = QVBoxLayout(self)
        
        self.btn_slot = QPushButton("Seleziona Immagine con Linea: Clicca qui, poi l'ImageWindow")
        self.btn_slot.clicked.connect(self.prepare_slot)
        layout.addWidget(self.btn_slot)
        self.show()

    def prepare_slot(self):
        # DISATTIVA I TOOL prima di entrare in selezione
        self.main_window.disable_all_tools()
        self.main_window.image_selection_callback = self.run_analysis
        self.btn_slot.setText("In attesa... Clicca una ImageWindow con Linea")

    def run_analysis(self, img, title, win_ref):
        if win_ref.line_coords is None:
            QMessageBox.warning(self.main_window, "Linea assente", 
                                f"Nella finestra '{title}' non è presente una Linea.\n"
                                "Usa il tool Linea e riprova.")
            self.btn_slot.setText("Riprova Selezione...")
            return

        y, xmin, xmax = win_ref.line_coords
        line_data = win_ref.image_data[y, xmin:xmax]
        x_axis = np.arange(xmin, xmax)

        from main_gui import PlotWindow
        self.window = PlotWindow(
            self.main_window,
            data_plot=(line_data, x_axis),
            info=f"Profilo riga y:{y} da {title}",
            title="Line Profile Plot"
        )
        self.window.show()
        self.btn_slot.setText("Seleziona un'altra finestra...")

class CalculateMeanCommand(QDialog):
    """Calcola la media globale tramite selezione slot."""
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Calculate Mean")
        layout = QVBoxLayout(self)
        
        self.btn_slot = QPushButton("Seleziona Immagine/Rettangolo/Linea: Clicca qui, poi clicca una ImageWindow")
        self.btn_slot.clicked.connect(self.prepare_slot)
        layout.addWidget(self.btn_slot)
        self.show()

    def prepare_slot(self):
        # DISATTIVA I TOOL prima di entrare in selezione
        self.main_window.disable_all_tools()
        self.main_window.image_selection_callback = self.run_analysis
        self.btn_slot.setText("In attesa... Clicca una ImageWindow")

    def run_analysis(self, img, title, win_ref):
        val = np.mean(img)
        QMessageBox.information(self.main_window, "Mean Value", 
                                f"Immagine: {title}\nValore Medio: {val:.6f}")
        self.btn_slot.setText("Seleziona Immagine (Cambia...)")


class CalculateStdCommand(QDialog):
    """Calcola la deviazione standard tramite selezione slot."""
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Standard Deviation")
        layout = QVBoxLayout(self)
        
        self.btn_slot = QPushButton("Seleziona Immagine/Rettangolo/Linea: Clicca qui, poi clicca una ImageWindow")
        self.btn_slot.clicked.connect(self.prepare_slot)
        layout.addWidget(self.btn_slot)
        self.show()

    def prepare_slot(self):
        # DISATTIVA I TOOL prima di entrare in selezione
        self.main_window.disable_all_tools()
        self.main_window.image_selection_callback = self.run_analysis
        self.btn_slot.setText("In attesa... Clicca una ImageWindow")

    def run_analysis(self, img, title, win_ref):
        val = np.std(img)
        QMessageBox.information(self.main_window, "Standard Deviation", 
                                f"Immagine: {title}\nDeviazione Standard: {val:.6f}")
        self.btn_slot.setText("Seleziona Immagine (Cambia...)")


class HistogramCommand(QDialog):
    """Genera l'istogramma tramite selezione slot."""
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Histogram")
        layout = QVBoxLayout(self)
        
        self.btn_slot = QPushButton("Seleziona Immagine/Rettangolo/Linea: Clicca qui, poi clicca una ImageWindow")
        self.btn_slot.clicked.connect(self.prepare_slot)
        layout.addWidget(self.btn_slot)
        self.show()

    def prepare_slot(self):
        # DISATTIVA I TOOL prima di entrare in selezione
        self.main_window.disable_all_tools()
        self.main_window.image_selection_callback = self.run_analysis
        self.btn_slot.setText("In attesa... Clicca una ImageWindow")

    def run_analysis(self, img, title, win_ref):
        hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 1])
        from main_gui import PlotWindow
        self.window = PlotWindow(
            self.main_window, 
            data_plot=(hist, bins[:-1]), 
            info=f"Istogramma di {title}",
            title="Histogram",
            is_histogram=True
        )
        self.window.show()
        self.btn_slot.setText("Seleziona Immagine (Cambia...)")
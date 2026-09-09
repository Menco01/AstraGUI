import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QToolBar, QMenu, 
                             QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QWhatsThis)
from PyQt5.QtCore import Qt, QEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from DataGeneration import data_generation_menu
from Reconstruction import reconstruction_menu
from Reconstruction import fbp_lpp_menu_unet_2
from Reconstruction import fbp_lpp_menu_unet_3
from Denoise import denoise_menu
from Analysis import analysis_menu

class ImageWindow(QDialog):
    def __init__(self, main_window, image_data, title="Image Window"):
        super().__init__(main_window)
        self.main_window = main_window
        self.image_data = image_data.copy()
        self.setWindowTitle(title)
        self.setModal(False) # NON bloccante
        self.resize(600, 600)

        # Testo di aiuto
        self.help_text_image_window = (
            "<h3>Guida alla Scheda Immagini</h3>"
            "<p>In questa sezione è possibile:</p>"
            "<ul>"
            "<li><b>Disegna Forme:</b> Cliccando uno dei bottoni '[] Rettangolo' | '— Linea' è possibile disegnare quella forma cliccando e trascinando il mouse all'interno dell'immagine.</li>"
            "<li><b>Salva Immagine:</b> Cliccando il bottone è possibile salvare l'immagine corrente.</li>"
            "<li><b>Pulisci Forme:</b> Cliccando il bottone è possibile rimuovere Rettangoli e Linee disegnate all'interno dell'immagine.</li>"
            "<li><b>Zoom e Spostamento:</b> Usando la rotellina del mouse è possibile zoomare l'immagine. Tramite le frecce direzionali è possibile spostare l'immagine nella direzione desiderata.</li>"
            "</ul>"
        )

        self.roi_coords = None   # [ymin, ymax, xmin, xmax]
        self.line_coords = None  # [y, xmin, xmax]
        self.line_artist = None  # L'oggetto grafico della linea

        layout = QVBoxLayout(self)
        self.figure = Figure()

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus() # Prende il focus all'apertura

        self.ax = self.figure.add_subplot(111)
        
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("Salva Immagine")
        btn_save.setFocusPolicy(Qt.NoFocus)
        btn_save.clicked.connect(self.save_image)
        btn_clean = QPushButton("Pulisci Forme")
        btn_clean.setFocusPolicy(Qt.NoFocus)
        btn_clean.clicked.connect(self.clean_shapes)
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_clean)

        layout.addWidget(self.canvas)
        layout.addLayout(btn_layout)

        self.ax.imshow(self.image_data, cmap='gray')
        self.ax.axis('off')
        
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('key_press_event', self.on_key)

        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        
        self.rs = RectangleSelector(self.ax, self.on_rect_select,
                                    useblit=True, button=[1], minspanx=5, minspany=5,
                                    spancoords='pixels', interactive=True,
                                    props=dict(facecolor='none', edgecolor='red', linestyle='--', linewidth=2))
        is_rect_active = (self.main_window.active_tool == "rectangle")
        self.rs.set_active(is_rect_active)
        
        self.canvas.draw()

    def event(self, event):
        # Intercetta il momento in cui l'utente clicca il bottone "?"
        if event.type() == QEvent.EnterWhatsThisMode and self.isActiveWindow():
            # Questo ripristina all'istante il cursore a freccia normale.
            QWhatsThis.leaveWhatsThisMode()
            self.show_help_image_window()
            return True # Restituisce True per bloccare il cambio del cursore di default
        
        # Lascia che gli altri eventi vengano gestiti normalmente
        return super().event(event)

    def show_help_image_window(self):
        help_box = QMessageBox(self)
        help_box.setWindowTitle("Help")
        help_box.setTextFormat(Qt.RichText) # Permette di usare tag HTML come <b>, <p>, <ul>
        help_box.setText(self.help_text_image_window)
        help_box.setIcon(QMessageBox.Information)
        help_box.exec_()

    def update_selectors_state(self):
        is_rect = (self.main_window.active_tool == "rectangle")
        self.rs.set_active(is_rect)

    def on_mouse_press(self, event):
        """Gestisce il click iniziale per la sostituzione immediata delle forme."""
        if event.inaxes != self.ax:
            return

        # Se il tool attivo è la Linea
        if self.main_window.active_tool == "line":
            # Rimuove IMMEDIATAMENTE il rettangolo ROI
            if self.roi_coords:
                self.roi_coords = None
                self.rs.set_visible(False)
                self.canvas.draw_idle()
            
            self.press_x = event.xdata
            self.press_y = event.ydata

        # Se il tool attivo è il Rettangolo
        elif self.main_window.active_tool == "rectangle":
            # Rimuove IMMEDIATAMENTE la linea se presente
            if self.line_artist:
                self.line_artist.remove()
                self.line_artist = None
                self.line_coords = None
                self.canvas.draw_idle()

    def on_mouse_release(self, event):
        if self.main_window.active_tool != "line" or event.inaxes != self.ax:
            return
        
        # Pulizia precedente
        self.clean_shapes()
        
        y = int(round(self.press_y))
        x1 = int(round(self.press_x))
        x2 = int(round(event.xdata))
        
        h, w = self.image_data.shape
        y = np.clip(y, 0, h-1)
        xmin, xmax = sorted([np.clip(x1, 0, w-1), np.clip(x2, 0, w-1)])
        
        if xmax > xmin:
            self.line_coords = [y, xmin, xmax]
            # Disegno grafico
            self.line_artist = self.ax.hlines(y, xmin, xmax, colors='red', linewidth=2)
            self.canvas.draw_idle()

    def on_click(self, event):
        self.main_window.last_focused_window = self
        
        # Gestione cattura immagine
        if self.main_window.image_selection_callback:
            img = self.get_active_image()
            self.main_window.image_selection_callback(img, self.windowTitle(), self)
            self.main_window.image_selection_callback = None
            return

        if self.main_window.active_tool == "rectangle":
            self.rs.set_active(True)
        else:
            self.rs.set_active(False)
            

    def on_rect_select(self, eclick, erelease):
        # Se viene disegnato un rettangolo, puliamo la linea
        if self.line_artist:
            self.line_artist.remove()
            self.line_artist = None
            self.line_coords = None

        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.roi_coords = (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
        self.main_window.last_focused_window = self

    def clean_shapes(self):
        self.roi_coords = None
        self.line_coords = None
        if self.line_artist:
            self.line_artist.remove()
            self.line_artist = None
        self.rs.set_visible(False)
        self.canvas.draw_idle()

    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Salva", "", "PNG (*.png);;JPG (*.jpg)")
        if path:
            img_to_save = (self.image_data * 255).astype(np.uint8) if self.image_data.max() <= 1.0 else self.image_data
            cv2.imwrite(path, img_to_save)

    def get_active_image(self):
        """Ritorna la ROI (se presente) o la Linea (se presente) o l'immagine intera."""
        if self.roi_coords:
            x1, x2, y1, y2 = self.roi_coords
            h, w = self.image_data.shape
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            return self.image_data[y1:y2, x1:x2]
        elif self.line_coords:
            y, xmin, xmax = self.line_coords
            h, w = self.image_data.shape
            # Assicuriamoci che y e x siano entro i limiti e interi
            y_idx = int(np.clip(y, 0, h - 1))
            x_start = int(max(0, xmin))
            x_end = int(min(w, xmax))
            return self.image_data[y_idx, x_start:x_end]
        return self.image_data

    # --- Zoom / Pan ---
    def on_scroll(self, event):
        if event.inaxes != self.ax: 
            return
        
        # Fattore di zoom (1.2 = 20% a ogni scatto)
        base_scale = 1.2
        xdata, ydata = event.xdata, event.ydata
        
        if xdata is None or ydata is None: 
            return

        # Determina il fattore di scala
        if event.button == 'up':
            scale_factor = 1 / base_scale # Ingrandisce
        else:
            scale_factor = base_scale     # Rimpicciolisce

        # Ottieni i limiti attuali
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        # Calcola le nuove distanze dai limiti rispetto alla posizione del mouse
        # Questo metodo preserva l'ordine (es. 512, 0) senza flippare l'asse
        new_xlim = [
            xdata - (xdata - cur_xlim[0]) * scale_factor,
            xdata + (cur_xlim[1] - xdata) * scale_factor
        ]
        new_ylim = [
            ydata - (ydata - cur_ylim[0]) * scale_factor,
            ydata + (cur_ylim[1] - ydata) * scale_factor
        ]

        # Applica i nuovi limiti
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        
        self.canvas.draw_idle()

    def on_key(self, event):
        if event.key is None: return
        
        cur_xlim = list(self.ax.get_xlim())
        cur_ylim = list(self.ax.get_ylim())
        
        # Calcoliamo lo spostamento come 10% della vista attuale
        # Usiamo abs() per essere sicuri dello step indipendentemente dall'orientamento
        width = abs(cur_xlim[1] - cur_xlim[0])
        height = abs(cur_ylim[1] - cur_ylim[0])
        step_x = width * 0.1
        step_y = height * 0.1

        # Direzioni
        if event.key == 'right':
            self.ax.set_xlim([cur_xlim[0] + step_x, cur_xlim[1] + step_x])
        elif event.key == 'left':
            self.ax.set_xlim([cur_xlim[0] - step_x, cur_xlim[1] - step_x])

        elif event.key == 'up':
            # "Su" significa diminuire i valori se lo 0 è in alto
            if cur_ylim[0] > cur_ylim[1]: # Caso standard imshow (Y invertita)
                self.ax.set_ylim([cur_ylim[0] - step_y, cur_ylim[1] - step_y])
            else: # Caso cartesiano standard
                self.ax.set_ylim([cur_ylim[0] + step_y, cur_ylim[1] + step_y])
                
        elif event.key == 'down':
            if cur_ylim[0] > cur_ylim[1]: # Caso standard imshow
                self.ax.set_ylim([cur_ylim[0] + step_y, cur_ylim[1] + step_y])
            else: # Caso cartesiano standard
                self.ax.set_ylim([cur_ylim[0] - step_y, cur_ylim[1] - step_y])

        self.canvas.draw_idle()


class PlotWindow(QDialog):
    """Usato per output plot con possibilità di click -> apri ImageWindow."""
    def __init__(self, main_window, images_dict=None, data_plot=None, title="Risultato", info="", is_histogram=False):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(1000, 450) # Leggermente più grande per ospitare i controlli
        
        # Testo di aiuto
        self.help_text_plot_window = (
            "<h3>Guida alla Scheda Plot</h3>"
            "<p>In questa sezione è possibile:</p>"
            "<ul>"
            "<li><b>Clicca Plot:</b> Cliccando sul plot o su una delle immagini nel plot, è possibile aprire una Scheda Immagine con l'immagine/plot selezionato.</li>"
            "</ul>"
        )

        layout = QVBoxLayout(self)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        if info: 
            lbl = QLabel(info)
            lbl.setWordWrap(True)
            layout.addWidget(lbl)

        # Se sono immagini da confrontare (2D)
        if images_dict:
            axes = self.figure.subplots(1, len(images_dict))
            if len(images_dict) == 1: axes = [axes]
            for ax, (name, img) in zip(axes, images_dict.items()):
                aspect = 'auto' if "sinogram" in name.lower() else 'equal'
                ax.imshow(img, cmap='gray', aspect=aspect)
                ax.set_title(name)
                ax.axis('off')
        
        # Se è un plot 1D (Mean / Histogram)
        elif data_plot:
            ax = self.figure.add_subplot(111)
            y_data, x_data = data_plot
            
            if is_histogram and x_data is not None:
                # Disegna l'istogramma a barre (rettangoli)
                ax.bar(x_data, y_data, width=np.diff(x_data)[0], color='gray', align='edge')
            else:
                # Disegna un grafico a linea continua
                if x_data is not None: 
                    ax.plot(x_data, y_data, color='blue', linewidth=1.5)
                else: 
                    ax.plot(y_data, color='blue', linewidth=1.5)
            
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_title(title)

        # Connessione dell'evento click
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)

    def event(self, event):
        # Intercetta il momento in cui l'utente clicca il bottone "?"
        if event.type() == QEvent.EnterWhatsThisMode and self.isActiveWindow():
            # Questo ripristina all'istante il cursore a freccia normale.
            QWhatsThis.leaveWhatsThisMode()
            self.show_help_plot_window()
            return True # Restituisce True per bloccare il cambio del cursore di default
        
        # Lascia che gli altri eventi vengano gestiti normalmente
        return super().event(event)

    def show_help_plot_window(self):
        help_box = QMessageBox(self)
        help_box.setWindowTitle("Help")
        help_box.setTextFormat(Qt.RichText) # Permette di usare tag HTML come <b>, <p>, <ul>
        help_box.setText(self.help_text_plot_window)
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
            
        # CASO 2: L'asse contiene grafici (plot a linee o istogrammi a barre)
        # Controllo se ci sono linee (event.inaxes.lines) o patch rettangolari (event.inaxes.patches, usate da ax.bar)
        elif len(event.inaxes.lines) > 0 or len(event.inaxes.patches) > 0:
            # "Fotografiamo" il grafico trasformandolo in un'immagine RGB
            self.canvas.draw()
            width, height = self.canvas.get_width_height()
            
            # Recuperiamo il buffer RGB dal canvas di Matplotlib
            buffer = np.frombuffer(self.canvas.tostring_rgb(), dtype=np.uint8)
            img_plot = buffer.reshape(height, width, 3)
            
            # Normalizziamo a 0-1
            img_plot = img_plot.astype(np.float32) / 255.0
            
            title = f"Snapshot: {self.windowTitle()}"
            self.main_window.open_image_window(img_plot, title)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASTRA GUI")
        self.resize(800, 200)
        self.setAcceptDrops(True)

        self.active_tool = None
        self.last_focused_window = None
        self.last_fbp_gt = None
        self.last_fbp_result = None
        self.image_selection_callback = None

        self.init_ui()

    def init_ui(self):
        menubar = self.menuBar()
        
        menu_data = menubar.addMenu("Data Generation")
        act_shepp = QAction("Genera Shepp-Logan", self)
        act_shepp.triggered.connect(lambda: data_generation_menu.SheppLoganDialog(self).show())
        menu_data.addAction(act_shepp)

        menu_rec = menubar.addMenu("Reconstruction")
        act_fbp = QAction("Filter Back Projection", self)
        act_fbp.triggered.connect(lambda: reconstruction_menu.FBPDialog(self).show())
        menu_rec.addAction(act_fbp)

        fbp_lpp_unet_2_action = QAction("FBP + LPP Denoising (U-Net 2)", self)
        fbp_lpp_unet_2_action.triggered.connect(lambda: fbp_lpp_menu_unet_2.FBPLPPDialog(self).show())
        menu_rec.addAction(fbp_lpp_unet_2_action)

        fbp_lpp_unet_3_action = QAction("FBP + LPP Denoising (U-Net 3)", self)
        fbp_lpp_unet_3_action.triggered.connect(lambda: fbp_lpp_menu_unet_3.FBPLPPDialog(self).show())
        menu_rec.addAction(fbp_lpp_unet_3_action)

        menu_denoise = menubar.addMenu("Denoise")
        act_denoise = QAction("Image Denoise", self)
        act_denoise.triggered.connect(lambda: denoise_menu.DenoiseDialog(self).show())
        menu_denoise.addAction(act_denoise)

        menu_analysis = menubar.addMenu("Analysis")

        act_diff = QAction("Image Difference", self)
        act_diff.triggered.connect(lambda: analysis_menu.ImageDifferenceDialog(self).show())

        plot_act = QAction("Plot", self)
        plot_act.triggered.connect(lambda: analysis_menu.PlotCommand(self))

        act_mean = QAction("Calculate Mean", self)
        act_mean.triggered.connect(lambda: analysis_menu.CalculateMeanCommand(self))

        act_std = QAction("Calculate Standard Deviation", self)
        act_std.triggered.connect(lambda: analysis_menu.CalculateStdCommand(self))

        act_hist = QAction("Istogramma", self)
        act_hist.triggered.connect(lambda: analysis_menu.HistogramCommand(self))

        menu_analysis.addActions([act_diff, plot_act, act_mean, act_std, act_hist])

        # Hover per Menu
        for menu in [menu_data, menu_rec, menu_denoise, menu_analysis]:
            menu.installEventFilter(self)

        # Toolbar Forme
        toolbar = QToolBar("Tools")
        self.addToolBar(toolbar)
        self.btn_rect = QAction("[ ] Rettangolo", self)
        self.btn_rect.setCheckable(True)
        self.btn_rect.triggered.connect(self.toggle_rect_tool)
        toolbar.addAction(self.btn_rect)

        toolbar.addSeparator() # Aggiunge la linea di separazione "|"

        self.btn_line = QAction("— Linea", self)
        self.btn_line.setCheckable(True)
        self.btn_line.triggered.connect(self.toggle_line_tool)
        toolbar.addAction(self.btn_line)

        lbl_info = QLabel("Trascina un'immagine in questa finestra per aprirla.")
        lbl_info.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(lbl_info)

    def disable_all_tools(self):
        """Disattiva i toggle dei tool Rettangolo e Linea e aggiorna le finestre."""
        self.active_tool = None
        self.btn_rect.setChecked(False)
        self.btn_line.setChecked(False)
        
        # Forza l'aggiornamento dello stato in tutte le ImageWindow aperte
        for dialog in self.findChildren(QDialog):
            if type(dialog).__name__ == "ImageWindow":
                if hasattr(dialog, 'update_selectors_state'):
                    dialog.update_selectors_state()

    def eventFilter(self, obj, event):
        if isinstance(obj, QMenu) and event.type() == QEvent.Enter:
            obj.popup(self.mapToGlobal(self.menuBar().actionGeometry(obj.menuAction()).bottomLeft()))
            return True
        return super().eventFilter(obj, event)

    def toggle_line_tool(self, checked):
        if checked:
            self.btn_rect.setChecked(False)
            self.active_tool = "line"
        else:
            self.active_tool = None
        
        for dialog in self.findChildren(QDialog):
            if type(dialog).__name__ == "ImageWindow":
                dialog.update_selectors_state()

    def toggle_rect_tool(self, checked):
        if checked:
            self.btn_line.setChecked(False)
            self.active_tool = "rectangle"
        else:
            self.active_tool = None
        
        for dialog in self.findChildren(QDialog):
            if type(dialog).__name__ == "ImageWindow":
                dialog.update_selectors_state()

    def disable_rectangle_tool(self):
        # Forza la disattivazione del tool rettangolo, spegnendo anche il bottone
        if self.btn_rect.isChecked():
            self.btn_rect.setChecked(False)
            self.toggle_rect_tool(False)

    def open_image_window(self, image_data, title="Image"):
        win = ImageWindow(self, image_data, title)
        win.show()
        self.last_focused_window = win

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            img = cv2.imread(url.toLocalFile(), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = img.astype(np.float32) / 255.0
                self.open_image_window(img, url.fileName())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
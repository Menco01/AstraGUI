from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QSpinBox, QLabel
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import numpy as np

class SheppLoganDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Genera Shepp-Logan")
        self.setModal(False)
        
        layout = QVBoxLayout(self)
        box = QHBoxLayout()
        box.addWidget(QLabel("Dimensione (pixel):"))
        self.spin = QSpinBox()
        self.spin.setRange(64, 2048)
        self.spin.setValue(512)
        box.addWidget(self.spin)
        
        btn_box = QHBoxLayout()
        btn_gen = QPushButton("Genera")
        btn_annulla = QPushButton("Annulla")
        btn_gen.clicked.connect(self.generate)
        btn_annulla.clicked.connect(self.close)
        btn_box.addWidget(btn_gen); btn_box.addWidget(btn_annulla)
        
        layout.addLayout(box)
        layout.addLayout(btn_box)

    def generate(self):
        size = self.spin.value()
        base = shepp_logan_phantom()
        img = resize(base, (size, size), order=1).astype(np.float32)
        self.main_window.open_image_window(img, f"Shepp-Logan {size}x{size}")
        self.close()
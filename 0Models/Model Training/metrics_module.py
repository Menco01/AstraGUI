import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

def calculate_PSNR(img_gt, img_rec):
    # Se le immagini sono identiche, MSE è 0 e PSNR tende a infinito
    if np.array_equal(img_gt, img_rec):
        return float('inf')
    
    data_range = img_gt.max() - img_gt.min()
    if data_range == 0: data_range = 1.0 # Evita divisione per zero se l'immagine è piatta
    
    return peak_signal_noise_ratio(img_gt, img_rec, data_range=data_range)

def calculate_SSIM(img_gt, img_rec):
    data_range = img_gt.max() - img_gt.min()
    return structural_similarity(img_gt, img_rec, data_range=data_range)

def calculate_MSE(img_gt, img_rec):
    return mean_squared_error(img_gt, img_rec)
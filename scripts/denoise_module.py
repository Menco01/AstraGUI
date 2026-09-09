import numpy as np
import cv2
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means, estimate_sigma
from scipy.ndimage import median_filter

def image_median_filter_denoise(image, size=3):
    return median_filter(image, size=size)

def image_bilateral_filter_denoise(image, d=9, sigma_color=75, sigma_space=75):
    # Opencv richiede float32 o uint8
    img_float = np.float32(image)
    denoised = cv2.bilateralFilter(img_float, d, sigma_color, sigma_space)
    return denoised

def image_total_variation_denoise(image, weight=0.1):
    return denoise_tv_chambolle(image, weight=weight)

def image_non_local_means_denoise(image):
    # Stima del sigma per NLM
    sigma_est = np.mean(estimate_sigma(image, channel_axis=None))
    denoised = denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True, 
                                patch_size=5, patch_distance=3, channel_axis=None)
    return denoised
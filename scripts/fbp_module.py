import numpy as np
import astra
from scipy.ndimage import gaussian_filter1d

def image_FBP(image, num_angles, sampling_density=1.0, cutoff_frequency=1.0):

    # Ottieni dimensioni reali: rows (altezza/y) e cols (larghezza/x)
    rows, cols = image.shape
    
    # Calcolo angoli
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    
    # Gestione Densità di Campionamento:
    # Aumentiamo il numero di detector e riduciamo la loro spaziatura 
    # per mantenere la stessa area di copertura (diagonale dell'immagine).
    base_detector_count = int(np.ceil(np.sqrt(rows**2 + cols**2)))
    detector_count = int(base_detector_count * sampling_density)
    detector_spacing = 1.0 / sampling_density
    
    # Creazione delle geometrie
    vol_geom = astra.create_vol_geom(rows, cols)
    # Inseriamo la densità tramite la spaziatura dei detector
    proj_geom = astra.create_proj_geom('parallel', detector_spacing, detector_count, angles)
    
    # Creazione del proiettore
    proj_id = astra.create_projector('linear', proj_geom, vol_geom)
    
    # Generazione del sinogramma
    sinogram_id, sinogram = astra.create_sino(image, proj_id)
    
    # Gestione Frequenza di Taglio (Cutoff Frequency):
    # Poiché ASTRA applica il filtro Ram-Lak internamente, applichiamo un pre-filtraggio 
    # passa-basso al sinogramma per limitare la banda passante.
    if cutoff_frequency < 1.0:
        # Mapping euristico: minore è la frequenza di taglio, maggiore è lo smoothing (sigma)
        # Se cutoff=1.0 -> sigma=0 (nessun filtro). Se cutoff=0.1 -> sigma elevato.
        f_c = max(1e-3, cutoff_frequency)
        sigma = (1.0 / f_c) - 1.0
        sinogram = gaussian_filter1d(sinogram, sigma=sigma, axis=1)
        # Sostituiamo il dato modificato nell'ID di ASTRA
        astra.data2d.store(sinogram_id, sinogram)
    
    # Configurazione FBP
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP')
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectorId'] = proj_id
    cfg['option'] = {'FilterType': 'Ram-Lak'}
    
    # Esecuzione algoritmo
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    
    reconstruction = astra.data2d.get(rec_id)
    
    # Pulizia delle risorse ASTRA
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)
    
    return sinogram, reconstruction
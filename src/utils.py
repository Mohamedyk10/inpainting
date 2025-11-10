
import numpy as np
import cv2
from scipy import ndimage

"""Tout ce qui nous sera utile pour le fichier principal"""

""" def make_patch(p, source_region, patch_size=9):
    half= patch_size//2
    patch = np.zeros((patch_size, patch_size, 3))
    for i in range(-half, +half+1):
        for j in range(-half, +half+1):
            patch[i+half, j+half]=source_region[(p[0]+i,p[1]+j)]
    return patch """
# Calculate values
def calculate_confidence(center, confidence_values, patch_size):
    """
    Calcule le terme de confiance C(p) pour un patch centré sur 'center'.
    C(p) = Somme des C(q) dans le patch / (taille du patch)²
    """
    i, j = center
    half = patch_size // 2
    
    total_confidence = 0.0
    
    for row in range(i - half, i + half + 1):
        for col in range(j - half, j + half + 1):

            if (row, col) in confidence_values:
                total_confidence += confidence_values[(row, col)]

    return total_confidence / (patch_size ** 2)

def calculate_dataterm2(center, source_region, target_region):
    """
    Calcule le terme de données D(p) en utilisant les trois canaux.
    Inclut un lissage gaussien initial pour atténuer l'effet du 'faux bord' du trou.
    """
    x,y = center
    alpha = 1.0
    
    # Conversion en Noir et blanc + gaussienne
    bw_img = 0.299 * source_region[:, :, 0] + 0.587 * source_region[:, :, 1] + 0.114 * source_region[:, :, 2] #à corriger peut-être
    bw_img = ndimage.uniform_filter(bw_img, size=3)
    #bw_img = ndimage.gaussian_filter(bw_img, sigma=0.5)

    # bw_img = ndimage.median_filter(bw_img, size=3) <- mieux pour le triangle mais très long
    # Calcul de la Normale (n_p)

    grad_mask_y, grad_mask_x = np.gradient(target_region.astype(np.float32))
    n_p = np.array([grad_mask_y[x,y], grad_mask_x[x,y]])
    norm_n_p = np.linalg.norm(n_p)
    if norm_n_p == 0:
        return 0.0
    n_p = n_p / norm_n_p
    

    # Itération sur les canaux R, G, B
    grad_I_y, grad_I_x = np.gradient(bw_img)
    
    # Le gradient (dx, dy) au point p pour ce canal
    isophote = np.array([-grad_I_y[x, y], grad_I_x[x, y]])
    
    # Normalisation
    # norme_isophote = np.linalg.norm(isophote_I_best)
    # if norme_isophote:
    #     normalise_isophote = isophote_I_best/norme_isophote 
    # else: normalise_isophote = isophote_I_best
    # Calcul de D(p)
    data_term = np.abs(np.dot(isophote, n_p)) / alpha
    
    return data_term

def calculate_dataterm(center, source_region, target_region):
    """
    Calcule D(p) = |(grad I)_perp . n_p| / alpha, en utilisant le canal 
    avec la magnitude de gradient la plus forte (vectorisé).
    """
    x, y = center
    alpha=1.0
    
    # Calculating n_p
    grad_mask_y, grad_mask_x = np.gradient(target_region.astype(np.float32))
    n_p_vector = np.array([grad_mask_y[x, y], grad_mask_x[x, y]]) 
    
    norm_n_p = np.linalg.norm(n_p_vector)
    if norm_n_p == 0:
        return 0.0
    n_p = n_p_vector / norm_n_p 
    
    
    # Calculating the isophote
    smoothed_region = ndimage.gaussian_filter(source_region, sigma=1.0)
    
    grad_y, grad_x = np.gradient(smoothed_region, axis=(0, 1))
    #print(f"grad_x : {grad_x.shape}; grad_y : {grad_y.shape}")
    # Extraction des 3 gradients (dy, dx) au point p = (x, y)
    # gradients_p.shape = (3, 2) où chaque ligne est [dy, dx] pour un canal
    gradients_p = np.array([
        [grad_y[x, y, 0], grad_x[x, y, 0]],  # Canal 0 (R)
        [grad_y[x, y, 1], grad_x[x, y, 1]],  # Canal 1 (G)
        [grad_y[x, y, 2], grad_x[x, y, 2]]   # Canal 2 (B)
    ])
    
    # Calcul des magnitudes de gradient pour les 3 canaux
    # magnitudes.shape = (3,)
    magnitudes = np.linalg.norm(gradients_p, axis=1)
    
    # Trouver l'indice du canal qui a la magnitude maximale
    best_channel_index = np.argmax(magnitudes)
    
    # Gradient du canal gagnant (vecteur [dy, dx])
    best_grad_I = gradients_p[best_channel_index]
    
    # Le vecteur Isophote I_perp est la rotation de 90° du gradient ([-dy, dx])
    isophote_I_best = np.array([-best_grad_I[0], best_grad_I[1]]) 
            
    dot_product = np.dot(isophote_I_best, n_p)
    data_term = np.abs(dot_product) / alpha
    
    return data_term

def make_patch(center, region, patch_size=9):
    "Retourne un patch centré sur un pixel (i,j)"
    i, j = center
    half = patch_size // 2
    return region[i - half:i + half + 1, j - half:j + half + 1]

def determine_closest_patch(target_region, patches : dict, contour_patch, p):
    # A modifier
    patch_p = contour_patch[p]; half = len(patch_p)//2
    mask_patch = make_patch(p, target_region, len(patch_p))
    existant_pixels = np.where(mask_patch==0)

    patch_p_mini = patch_p[existant_pixels].flatten()

    patch_matrix = np.array([patches[q][existant_pixels].flatten() for q in patches])
    distances = np.linalg.norm(patch_matrix-patch_p_mini, axis=1)
    min_index = np.argmin(distances)
    keys_patches = list(patches.keys())
    return keys_patches[min_index]

"""
ou bien

def determine_closest_patch(source_patches, target_patch):
    "Trouve le patch le plus proche dans la source"
    best_dist = float('inf')
    best_patch = None

    for key, src_patch in source_patches.items():
        if src_patch.shape != target_patch.shape:
            continue
        dist = np.sum((src_patch - target_patch) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_patch = key
    return best_patch
"""

def add_mask_rect(image):
    if image.ndim == 3:
        hauteur, largeur, nb_chaines = image.shape
    else:
        hauteur, largeur = image.shape
    
    print(f"Dimensions de l'image : {hauteur} x {largeur}")
    
    
    x1 = int(input("Entrer x1 (colonne gauche) : "))
    y1 = int(input("Entrer y1 (ligne haute)   : "))
    x2 = int(input("Entrer x2 (colonne droite): "))
    y2 = int(input("Entrer y2 (ligne basse)   : "))
    
    
    masque = np.zeros((hauteur, largeur), dtype=np.uint8)
    masque[y1:y2, x1:x2] = 1
    
    return masque

'''
masque = ajouter_mask_rect()
plt.imshow(masque, cmap="gray")
plt.show()
'''

def get_image_name(filename):
    name = ""
    for c in filename:
        name += c
        if c==".":
            break
    return name + "inpainted.webp"

import numpy as np
from scipy import ndimage

"""Tout ce qui nous sera utile pour le fichier principal"""

# Calculate values
def calculate_confidence(center, confidence_values, patch_size):
    """
    Calculate the confidence term C(p) for a centered patch around ``center``.

    C(p) = (sum of C(q) of the pixels q from the patch) / patch_size²
    """
    i, j = center
    half = patch_size // 2
    
    total_confidence = 0.0
    
    for row in range(i - half, i + half + 1):
        for col in range(j - half, j + half + 1):

            if (row, col) in confidence_values:
                total_confidence += confidence_values[(row, col)]

    return total_confidence / (patch_size ** 2)

def calculate_dataterm(center, source_region, target_region):
    """
    Calculate the data term using the maximal isophote of the three RGB canals. 

    **Very slow method.**
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
    gradients_p = np.array([
        [grad_y[x, y, 0], grad_x[x, y, 0]],  # Canal 0 (R)
        [grad_y[x, y, 1], grad_x[x, y, 1]],  # Canal 1 (G)
        [grad_y[x, y, 2], grad_x[x, y, 2]]   # Canal 2 (B)
    ])
    
    magnitudes = np.linalg.norm(gradients_p, axis=1)
    best_channel_index = np.argmax(magnitudes)
    best_grad_I = gradients_p[best_channel_index]
    
    # we need the perpendicular of the isophote
    isophote_I_best = np.array([-best_grad_I[0], best_grad_I[1]]) 
            
    dot_product = np.dot(isophote_I_best, n_p)
    data_term = np.abs(dot_product) / alpha
    
    return data_term

def calculate_dataterm2(center, source_region, target_region):
    """
    Calculate dataterm D(p) by transforming the image into a gray-scale image.
    We use a smoothing filter in order to correct the effect of "edges" in the border with target region

    **Faster method, it is the method used in our final version.**
    """
    x,y = center
    alpha = 1.0
    
    # Conversion en Noir et blanc + gaussienne
    bw_img = 0.299 * source_region[:, :, 0] + 0.587 * source_region[:, :, 1] + 0.114 * source_region[:, :, 2] #à corriger peut-être
    #bw_img = ndimage.uniform_filter(bw_img, size=3)
    bw_img = ndimage.gaussian_filter(bw_img, sigma=0.5)

    # bw_img = ndimage.median_filter(bw_img, size=3) <- mieux pour le triangle mais très long

    # n_p computing
    grad_mask_y, grad_mask_x = np.gradient(target_region.astype(np.float32))
    n_p = np.array([grad_mask_y[x,y], grad_mask_x[x,y]])
    norm_n_p = np.linalg.norm(n_p)
    if norm_n_p == 0:
        return 0.0
    n_p = n_p / norm_n_p
    

    # Itération sur les canaux R, G, B
    grad_I_y, grad_I_x = np.gradient(bw_img)
    
    # We should have it perpendicular to the gradient
    isophote = np.array([-grad_I_y[x, y], grad_I_x[x, y]])
    
    # Final Answer
    data_term = np.abs(np.dot(isophote, n_p)) / alpha
    
    return data_term

def make_patch(center, region, patch_size=9):
    "Returns a centered patch around center=(i,j)"
    i, j = center
    half = patch_size // 2
    return region[i - half:i + half + 1, j - half:j + half + 1]

def determine_closest_patch(target_region, patches : dict, contour_patch, p):
    """Determines the closest patch from the dict. of ``patches`` to the centered patch around ``p``."""
    patch_p = contour_patch[p]; half = len(patch_p)//2
    mask_patch = make_patch(p, target_region, len(patch_p))
    existant_pixels = np.where(mask_patch==0)

    patch_p_mini = patch_p[existant_pixels].flatten()

    patch_matrix = np.array([patches[q][existant_pixels].flatten() for q in patches])
    distances = np.linalg.norm(patch_matrix-patch_p_mini, axis=1)
    min_index = np.argmin(distances)
    keys_patches = list(patches.keys())
    return keys_patches[min_index]

def add_mask_rect(image):
    """Creates a rectangular mask"""
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

def get_image_name(filename):
    name = ""
    for c in filename:
        name += c
        if c==".":
            break
    return name + "inpainted.webp"
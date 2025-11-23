
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
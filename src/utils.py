
import numpy as np
import cv2

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
    half = patch_size//2
    confidences = [confidence_values[i-half+center[0], j-half+center[0]] for i in range(half) for j in range(half)]
    return sum(confidences)/patch_size**2

def calculate_dataterm(center):
    pass

def make_patch(center, source_region, patch_size=9):
    "Retourne un patch centré sur un pixel (i,j)"
    i, j = center
    half = patch_size // 2
    return source_region[i - half:i + half + 1, j - half:j + half + 1]

def determine_closest_patch(target_region, patches : dict, contour_patch, p):
    # A modifier
    patch_p = contour_patch[p]; half = len(patch_p)//2
    min_dist = float("inf")
    min_index = 0,0
    mask_patch = make_patch(p, target_region, len(patch_p))
    existant_pixels = np.where(mask_patch==0)

    patch_p_mini = patch_p[existant_pixels]
    for q, patch_q in patches.items():
        patch_q_mini = patch_q[existant_pixels]
        dist = np.linalg.norm(patch_p_mini-patch_q_mini)
        if dist<min_dist:
            min_dist=dist
            min_index=q
    return min_index

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

def gradient(f):
    pass

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

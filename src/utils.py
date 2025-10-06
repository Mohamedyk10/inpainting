
import numpy as np
import cv2

"""Tout ce qui nous sera utile pour le fichier principal"""

def extract_patch(image, center, patch_size):
    "Retourne un patch centré sur un pixel (i,j)"
    i, j = center
    half = patch_size // 2
    return image[i - half:i + half + 1, j - half:j + half + 1, :]

def determine_closest_patch(patches : dict, p):
    # A modifier
    patch_p = patches[p]
    return np.argmin(np.linalg.norm(patch_p-patches[q]) for q in patches.keys if q != p)

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

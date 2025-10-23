
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
    """
    Calcule le terme de confiance C(p) pour un patch centré sur 'center'.
    C(p) = Somme des C(q) dans le patch / (taille du patch)²
    """
    i, j = center
    half = patch_size // 2
    
    # 1. Extraire la zone de confiance correspondant au patch
    #    La variable 'confidence_values' est un dictionnaire {(i,j): float}
    
    total_confidence = 0.0
    
    # Parcourir les coordonnées du patch
    for row in range(i - half, i + half + 1):
        for col in range(j - half, j + half + 1):
            # S'assurer que les coordonnées sont valides (dans les limites de l'image)
            if (row, col) in confidence_values:
                total_confidence += confidence_values[(row, col)]
            # Note : on pourrait ajouter une gestion des bords si (row, col) sort des limites, 
            # mais nous supposons que les centres de patchs sont initialisés correctement.

    # 2. Diviser par la surface totale du patch
    return total_confidence / (patch_size ** 2)

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

def get_image_name(filename):
    name = ""
    for c in filename:
        name += c
        if c==".":
            break
    return name + "inpainted.webp"
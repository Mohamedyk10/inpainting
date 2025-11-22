import numpy as np
import cv2
from scipy import ndimage

# --- 1. FONCTIONS DE GESTION DES GRADIENTS (NOUVEAU) ---

def initialize_gradients(source_region, sigma_lissage):
    """
    Calcule les gradients initiaux sur toute l'image.
    Retourne des listes contenant les gradients Y et X pour chaque canal.
    """
    # Lissage global initial
    smoothed_region = ndimage.gaussian_filter(source_region, sigma=(sigma_lissage, sigma_lissage, 0))
    
    grads_y = []
    grads_x = []
    
    # Calcul pour chaque canal (R, G, B)
    for channel in range(smoothed_region.shape[2]):
        gy, gx = np.gradient(smoothed_region[:, :, channel])
        grads_y.append(gy)
        grads_x.append(gx)
        
    return grads_y, grads_x

def update_local_gradients(source_region, grads_y, grads_x, p, patch_size, sigma_lissage):
    """
    Met à jour les tableaux globaux de gradients UNIQUEMENT autour du patch p.
    C'est l'optimisation clé : on ne recalcule pas tout.
    """
    # Marge de sécurité pour le lissage (pour éviter les effets de bord du filtre)
    margin = 10 
    full_h, full_w = source_region.shape[:2]
    i, j = p
    
    # Définir la fenêtre de mise à jour (Patch + Marge)
    i_min = max(0, i - patch_size//2 - margin)
    i_max = min(full_h, i + patch_size//2 + margin + 1)
    j_min = max(0, j - patch_size//2 - margin)
    j_max = min(full_w, j + patch_size//2 + margin + 1)
    
    # Extraire la sous-région
    sub_source = source_region[i_min:i_max, j_min:j_max, :]
    
    # Appliquer le lissage sur cette PETITE sous-région
    sub_smoothed = ndimage.gaussian_filter(sub_source, sigma=(sigma_lissage, sigma_lissage, 0))
    
    # Recalculer les gradients pour chaque canal sur la sous-région
    for c in range(source_region.shape[2]):
        sub_gy, sub_gx = np.gradient(sub_smoothed[:, :, c])
        
        # Mettre à jour les tableaux globaux (copier-coller)
        # On écrase les anciennes valeurs avec les nouvelles
        grads_y[c][i_min:i_max, j_min:j_max] = sub_gy
        grads_x[c][i_min:i_max, j_min:j_max] = sub_gx

# --- 2. CALCUL DU DATA TERM (Lecture optimisée) ---

def calculate_dataterm_optimized(center, grad_mask_y, grad_mask_x, grads_y, grads_x):
    """
    Calcule D(p) en lisant simplement dans les tableaux globaux mis à jour.
    Extrêmement rapide.
    """
    i, j = center
    
    # 1. Normale (n_p)
    n_p = np.array([grad_mask_y[i, j], grad_mask_x[i, j]])
    norm_n_p = np.linalg.norm(n_p)
    if norm_n_p == 0: return 0.0
    n_p = n_p / norm_n_p

    # 2. Isophote (Lecture dans les tableaux globaux)
    max_grad_magnitude = 0.0 
    isophote_I_best = np.array([0.0, 0.0])
    
    # On itère sur les 3 canaux (R, G, B)
    for c in range(len(grads_y)):
        # Lecture directe à la position (i, j)
        gy = grads_y[c][i, j]
        gx = grads_x[c][i, j]
        
        grad_vec = np.array([gx, gy])
        mag = np.linalg.norm(grad_vec)
        
        if mag > max_grad_magnitude:
            max_grad_magnitude = mag
            isophote_I_best = np.array([-gy, gx]) # Orthogonal

    # 3. Calcul Final
    data_term = np.abs(np.dot(isophote_I_best, n_p)) / 1.0
    return min(data_term, 1.0)

# --- 3. AUTRES FONCTIONS UTILITAIRES (Inchangées) ---

def calculate_confidence(center, confidence_values, patch_size):
    i, j = center
    half = patch_size // 2
    total_confidence = 0.0
    for row in range(i - half, i + half + 1):
        for col in range(j - half, j + half + 1):
            if (row, col) in confidence_values:
                total_confidence += confidence_values[(row, col)]
    return total_confidence / (patch_size ** 2)

def make_patch(center, source_region, patch_size=9):
    i, j = center
    half = patch_size // 2
    return source_region[i - half:i + half + 1, j - half:j + half + 1]

def determine_closest_patch(target_region, patches: dict, contour_patch, p, search_radius):
    patch_p = contour_patch[p]
    mask_patch = make_patch(p, target_region, len(patch_p))
    existant_pixels = np.where(mask_patch == 0)
    patch_p_mini = patch_p[existant_pixels].flatten()

    p_i, p_j = p
    filtered_keys = []
    for q_key in patches.keys():
        q_i, q_j = q_key
        if abs(p_i - q_i) < search_radius and abs(p_j - q_j) < search_radius:
            filtered_keys.append(q_key)

    if not filtered_keys:
        filtered_keys = list(patches.keys())

    patch_matrix = np.array([patches[q_key][existant_pixels].flatten() for q_key in filtered_keys])
    distances = np.linalg.norm(patch_matrix - patch_p_mini, axis=1)
    min_index = np.argmin(distances)
    return filtered_keys[min_index]

def add_mask_rect(image):
    # (Votre code add_mask_rect inchangé)
    if image.ndim == 3: hauteur, largeur, _ = image.shape
    else: hauteur, largeur = image.shape
    print(f"Dimensions : {hauteur} x {largeur}")
    x1 = int(input("x1 (gauche) : "))
    y1 = int(input("y1 (haut)   : "))
    x2 = int(input("x2 (droite) : "))
    y2 = int(input("y2 (bas)    : "))
    masque = np.zeros((hauteur, largeur), dtype=np.uint8)
    masque[y1:y2, x1:x2] = 1
    return masque

def get_image_name(filename):
    name = ""
    for c in filename:
        name += c
        if c==".": break
    return name + "inpainted.webp"
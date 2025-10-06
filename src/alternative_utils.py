import numpy as np

def extract_patch(image, center, patch_size):
    """Retourne un patch centré sur un pixel (i,j)"""
    i, j = center
    half = patch_size // 2
    return image[i - half:i + half + 1, j - half:j + half + 1, :]

def determine_closest_patch(source_patches, target_patch):
    """Trouve le patch le plus proche dans la source"""
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

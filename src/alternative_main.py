import numpy as np
import cv2
import os
from scipy import ndimage
import matplotlib.pyplot as plt
from alternative_utils import *

# -----------------------------------------------------------
# 🔹 Chargement d'une image et d'un masque par nom de fichier
# -----------------------------------------------------------
def load_image_from_database(filename_original="6.original.webp", filename_mask="6.mask.webp"):
    # Chemin absolu du dossier 'data' (voisin de 'src')
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))

    # Construit les chemins complets
    image_path = os.path.join(data_dir, filename_original)
    mask_path  = os.path.join(data_dir, filename_mask)

    # Chargement
    image = plt.imread(image_path).copy()  # ✅ .copy() pour rendre modifiable
    mask  = plt.imread(mask_path)

    # Si le masque est RGB → on garde un canal
    if mask.ndim == 3:
        mask = mask[..., 0]

    # Binarisation
    mask = (mask > 0).astype(np.uint8)

    print(f"✅ Image chargée : {image_path}")
    print(f"✅ Masque chargé : {mask_path}")
    return image, mask


# -----------------------------------------------------------
# 🔹 Affichage
# -----------------------------------------------------------
def display_with_contour(image, mask, iteration):
    contour = mask - ndimage.binary_erosion(mask).astype(np.uint8)
    img_copy = image.copy()
    img_copy[contour == 1] = [1, 0, 0]  # rouge (R=1, G=0, B=0)
    plt.figure(figsize=(8,8))
    plt.imshow(img_copy)
    plt.title(f"Itération {iteration}")
    plt.axis("off")
    plt.show()

def display_image(image, title="Image"):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


# -----------------------------------------------------------
# 🔹 Calcul du contour du masque
# -----------------------------------------------------------
def get_contour(mask):
    return mask - ndimage.binary_erosion(mask).astype(np.uint8)


# -----------------------------------------------------------
# 🔹 Classe principale d'Inpainting
# -----------------------------------------------------------
class Inpainting:
    def __init__(self, patch_size=9, filename_original="6.original.webp", filename_mask="6.mask.webp"):
        self.image, self.mask = load_image_from_database(filename_original, filename_mask)
        self.patch_size = patch_size
        self.target_region = self.mask
        self.source_region = (1 - self.mask)[..., None] * self.image
        self.contour_region = get_contour(self.mask)
        self.contour = np.argwhere(self.contour_region == 1)
        print(f"Image shape: {self.image.shape}, {len(self.contour)} contour pixels")

    def update_patches(self):
        half = self.patch_size // 2
        self.source_patches = {
            (i, j): extract_patch(self.image, (i, j), self.patch_size)
            for i in range(half, self.image.shape[0] - half)
            for j in range(half, self.image.shape[1] - half)
            if self.mask[i, j] == 0
        }

    def best_match_patch(self, target_center):
        target_patch = extract_patch(self.image, target_center, self.patch_size)
        best_patch_key = determine_closest_patch(self.source_patches, target_patch)
        return best_patch_key

    def copy_patch(self, source_center, target_center):
        src_patch = extract_patch(self.image, source_center, self.patch_size)
        half = self.patch_size // 2
        i, j = target_center

        # Copier uniquement les pixels manquants (mask = 1)
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                y, x = i + di, j + dj
                if 0 <= y < self.image.shape[0] and 0 <= x < self.image.shape[1]:
                    if self.mask[y, x] == 1:
                        self.image[y, x, :] = src_patch[di + half, dj + half, :]
                        self.mask[y, x] = 0  # Pixel rempli

    def run(self):
        self.update_patches()
        iteration = 0
        while np.any(self.mask == 1):
            contour = get_contour(self.mask)
            contour_points = np.argwhere(contour == 1)
            if len(contour_points) == 0:
                break

            p = tuple(contour_points[len(contour_points) // 2])  # pixel du contour
            q = self.best_match_patch(p)
            self.copy_patch(q, p)

            iteration += 1
            if iteration % 5 == 0:
                #display_image(self.image, f"Iteration {iteration}")
                display_with_contour(self.image, self.mask, iteration)

        display_image(self.image, "Inpainting final")
        return self.image


# -----------------------------------------------------------
# 🔹 Programme principal
# -----------------------------------------------------------
if __name__ == "__main__":
    inpaint_process = Inpainting(patch_size=9, filename_original="6.original.webp", filename_mask="6.mask.webp")
    result = inpaint_process.run()

    # Sauvegarde du résultat
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "result.png")

    cv2.imwrite(output_path, cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"💾 Résultat sauvegardé dans : {output_path}")

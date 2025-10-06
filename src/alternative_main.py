import numpy as np
import cv2
import os
from glob import glob
from scipy import ndimage
import matplotlib.pyplot as plt
from utils import *

# Rendre le chemin robuste
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
original_filepaths = glob(os.path.join(data_dir, '*original.webp'))
mask_filepaths = glob(os.path.join(data_dir, '*mask.webp'))
current_img = 0


def load_image_from_database():
    global mask, image, current_img
    image = plt.imread(original_filepaths[current_img])
    mask = plt.imread(mask_filepaths[current_img])

    if mask.ndim == 3:
        mask = mask[..., 0]

    mask = (mask > 0).astype(np.uint8)
    if current_img < len(original_filepaths) - 1:
        current_img += 1
    return image, mask


def display_image(image, title="Image"):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


def get_contour(mask):
    return mask - ndimage.binary_erosion(mask).astype(np.uint8)


class Inpainting:
    def __init__(self, patch_size=9):
        self.image, self.mask = load_image_from_database()
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

        # Copier uniquement les pixels manquants
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                y, x = i + di, j + dj
                if 0 <= y < self.image.shape[0] and 0 <= x < self.image.shape[1]:
                    if self.mask[y, x] == 1:
                        self.image[y, x, :] = src_patch[di + half, dj + half, :]
                        self.mask[y, x] = 0

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
                display_image(self.image, f"Iteration {iteration}")

        display_image(self.image, "Inpainting final")
        return self.image


if __name__ == "__main__":
    inpaint_process = Inpainting(patch_size=9)
    result = inpaint_process.run()
    cv2.imwrite("output/result.png", cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

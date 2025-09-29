import pandas as pd
import numpy as np

from glob import glob

from scipy import ndimage
import matplotlib.pylab as plt

from utils import *

import os

# Rendre le chemin robuste quel que soit le dossier d'exécution
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
original_filepaths = glob(os.path.join(data_dir, '*original.webp'))
mask_filepaths = glob(os.path.join(data_dir, '*mask.webp'))
current_img=0

image = np.array([])
mask = np.array([])

def load_image_from_database():
    global mask, image, current_img
    image = plt.imread(original_filepaths[current_img])
    mask = plt.imread(mask_filepaths[current_img])
    if current_img<len(original_filepaths)-1:
        current_img+=1
    return image, mask

def display_image(image):
    fig, ax = plt.subplots(figsize= (10,10))
    ax.imshow(image)
    ax.axis('off')
    plt.show()

def targetify(masque):
    return np.array([255 if image[x][y]>128 else 0 for x in range(len(masque)) for y in range(len(masque[0]))])

def get_contour(target):
    return target-ndimage.binary_erosion(target)

def save_image(image_name, image):
    plt.imsave("output/"+image_name, image)

class Inpainting():
    def get_source_region(self):
        return np.array([self.image[x][y] if self.target_region[x][y]==0 else 0 for x in range(len(self.mask)) for y in range(len(self.mask[0]))])
    def __init__(self, image, mask):
        self.image, self.mask = load_image_from_database()
        self.target_region = targetify(self.mask)
        self.source_region = self.get_source_region()
        self.contour = get_contour(self.target_region)

        self.patches=np.array([])
        self.priority_patches = np.array([])
        
    def update_priority(self):
        pass

    def update_regions(self):
        pass

    def create_patches(self):
        """Create patches for every pixel in the contour"""
        pass

    def patch_to_use(self):
        """Return the patch with highest priority"""
        pass

    def best_match_sample(self):
        """Returns the best match patch"""
        pass
        
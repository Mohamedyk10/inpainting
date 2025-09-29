import pandas as pd
import numpy as np

from glob import glob

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

def save_image(image_name, image):
    plt.imsave("output/"+image_name, image)

class Inpainting():
    def __init__(self, image, mask):
        self.image, self.mask = load_image_from_database()
        self.source_region = np.array([])
        self.target_region = np.array([])
        self.contour = np.array([])

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
        
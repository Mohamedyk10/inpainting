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

    #Si le masque est RGB 
    if mask.ndim == 3:
        mask = mask[...,0]

    #Pour le rendre binaire
    mask = (mask > 0).astype(np.uint8)
    
    if current_img<len(original_filepaths)-1:
        current_img+=1
    return image, mask

def display_image(image):
    fig, ax = plt.subplots(figsize= (10,10))
    ax.imshow(image)
    ax.axis('off')
    plt.show()

'''def targetify(masque):
    return image * mask[..., None] '''

def get_contour(mask):
    return mask - ndimage.binary_erosion(mask).astype(np.uint8)

def save_image(image_name, image):
    plt.imsave("output/"+image_name, image)

class Inpainting():
    def get_source_region(self):
        return image * (1 - mask)[..., None] 
    def __init__(self):
        self.image, self.mask = load_image_from_database()
        # Regions 
        self.target_region = mask[..., None] #ou juste mask
        self.source_region = self.get_source_region()
        self.contour_region = get_contour(self.mask)
        # Contour elements (array)
        self.contour = np.argwhere(self.contour == 1) 

        # Patches initialization
        self.source_patches = {}
        self.contour_patches={} # {(i,j): patch}
        self.update_patches()

        # values for patches
        self.confidence_values = {(i,j): int(self.mask[i,j]==0) for i in range(len(self.mask)) for j in range(len(self.mask[1]))} #{(i,j) : priority}
        self.data_terms = {} #same
        self.priority_patches = {} #same
        print(self.mask.shape)

    def calculate_priority(self):
        pass

    def update_regions(self):
        pass

    def update_patches(self, patch_size=9):
        """Create a patch for a pixel in the contour"""
        half = patch_size//2
        self.contour_patches={(i,j): self.source_region[i-half:i+half, j-half:j+half] for (i,j) in self.contour if i-half >= 0 and i+half < self.image.shape[0] and j-half >= 0 and j+half < self.image.shape[1]}
        self.source_patches={p: self.source_region[p[0]-half:p[0]+half, p[1]-half:p[1]+half] for p in np.argwhere(self.mask==0)}

    def patch_to_use(self):
        """Return the patch with highest priority"""
        i = max(self.priority_patches,key=self.priority_patches.get)
        return self.contour_patches[i]

    def best_match_sample(self, p): # p = (i,j)
        """Returns the best match patch"""
        return determine_closest_patch(self.source_patches, p)

    def update_values(self,p,q):
        self.patches[p] = np.array([[self.patches[q][i,j] if self.mask[i,j] else self.patches[p][i,j] for i in range(len(self.patches[p]))] for j in range(len(patches[p][0]))])
        # Je sais pas si le code est idéal
        for i in range(len(self.contour_patches[p])):
            for j in range(len(self.source_patches[q])):
                if self.mask[i,j]:
                    self.confidence_values[i,j]=self.confidence_values[(q[0]+i-1, q[1]+j-1)]
                    # self.update_regions() ??????
        pass

    def display(self):
        display_image(self.source_region)
output = Inpainting()
output.display()


#Pour tester:

if __name__ == "__main__":
    img = cv2.imread("images/example_original.png")
    mask = cv2.imread("images/example_mask.png", 0)
    result = inpaint(img, mask, patch_size=9)
    cv2.imwrite("result.png", result)

import pandas as pd
import numpy as np
from glob import glob
from scipy import ndimage
import matplotlib.pylab as plt
from utils import *
import os
import random
import time

# Rendre le chemin robuste quel que soit le dossier d'exécution
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
original_filepaths = glob(os.path.join(data_dir, '*original.webp'))
mask_filepaths = glob(os.path.join(data_dir, '*mask.webp'))

'''def targetify(mask):
    return image * mask[..., None] '''


class Inpainting():
    """A class that implements Exemplar-based inpainting method"""

    """Loading data"""
    def load_image_from_database(self):
        self.image = plt.imread(original_filepaths[self.current_img])
        self.mask = plt.imread(mask_filepaths[self.current_img])

        #Si le masque est RGB 
        if self.mask.ndim == 3:
            self.mask = self.mask[...,0]

        #Pour le rendre binaire
        self.mask = (self.mask > 0).astype(np.uint8)
        
        if self.current_img<len(original_filepaths)-1:
            self.current_img+=1
    
    """Getters"""
    def get_contour(self):
        return self.target_region - ndimage.binary_erosion(self.target_region).astype(np.uint8)
    def get_source_region(self):
        return self.image * (1 - self.mask)[..., None] 
    def get_source_region(self):
        return self.image * (1 - self.mask)[..., None] 
    
    def __init__(self, patch_size=9, curr_im = 0):
        # Dataset related variables
        self.current_img=curr_im
        self.image = np.array([])
        self.mask = np.array([])
        self.load_image_from_database()

        if self.image.dtype == np.uint8 or np.nanmax(self.image) > 1.0:
            self.image = self.image.astype(np.float32) / 255.0

        # Regions 
        self.target_region = self.mask.copy()
        self.source_region = self.get_source_region()
        self.contour_region = self.get_contour()
        # Contour elements (array)
        self.contour = [(int(x), int(y)) for x, y in np.argwhere(self.contour_region == 1)]

        # Patches initialization
        self.source_patches = {}
        self.contour_patches={} # {(i,j): patch}
        self.patch_size = 9
        self.update_patches()

        # values for patches
        self.confidence_values = {(i,j): int(self.target_region[i,j]==0) for i in range(len(self.target_region)) for j in range(len(self.target_region[1]))} #{(i,j) : priority}
        self.data_terms = {} #same
        self.priority_patches = {} #same
    
    def calculate_priority(self):
        pass

    def update_regions(self, p):
        x,y = p
        i0, i1 = x-self.patch_size//2, x+self.patch_size//2+1
        j0, j1 = y-self.patch_size//2, y+self.patch_size//2+1
        self.target_region[i0:i1, j0:j1]=0
        self.source_region[i0:i1, j0:j1] = self.contour_patches[p]
        self.contour_region = self.get_contour()
        self.contour = [(int(x), int(y)) for x, y in np.argwhere(self.contour_region == 1)]
        pass

    def update_patches(self):
        """Create a patch for a pixel in the contour"""
        half = self.patch_size//2
        self.contour_patches = {(i, j): make_patch((i, j), self.source_region, self.patch_size) for (i, j) in self.contour if i-half >= 0 and i+half < self.image.shape[0] and j-half >= 0 and j+half < self.image.shape[1]}
        self.source_patches = {(i,j): make_patch((i,j), self.source_region, self.patch_size) for i in range(half, self.image.shape[0]-half) for j in range(half, self.image.shape[1]-half) if np.all(self.target_region[i-half:i+half+1, j-half:j+half+1]==0 )}

    def patch_to_use(self):
        """Return the key of the patch with highest priority"""
        return random.choice(list(self.contour_patches))
        i = max(self.priority_patches,key=self.priority_patches.get)
        return i

    def best_match_sample(self, p): # p = (i,j)
        """Returns the best match patch"""
        return determine_closest_patch(self.target_region, self.source_patches, self.contour_patches, p)

    def update_values(self,p,patch_q):
        half = self.patch_size//2
        i0, i1 = p[0] - half, p[0] + half + 1
        j0, j1 = p[1] - half, p[1] + half + 1
        target_values = self.target_region[i0:i1,j0:j1].copy()
        unknown_pixels = (target_values==1)
        new_patch_val = self.contour_patches[p].copy()
        new_patch_val[unknown_pixels]=patch_q[unknown_pixels]
        #self.contour_patches[p] = np.array([[self.source_patches[q][i,j] if self.mask[i,j] else self.contour_patches[p][i,j] for i in range(self.patch_size)] for j in range(len(self.patch_size))])
        self.contour_patches[p]=new_patch_val
        # Je sais pas si le code est idéal
        # for i in range(len(self.contour_patches[p])):
        #     for j in range(len(self.source_patches[q])):
        #         if self.mask[i,j]:
        #             self.confidence_values[i,j]=self.confidence_values[q[0]+i-1, q[1]+j-1]
        pass


    """Main Function"""
    def inpaint(self):
        num_iter = 0 # Juste pour le débuggage, à retirer ensuite
        while np.any(self.target_region==1) and num_iter<200:
            print("Iteration : " + str(num_iter))
            self.calculate_priority()
            p = self.patch_to_use()
            patch_p = self.contour_patches[p]
            q = self.best_match_sample(p); patch_q = self.source_patches[q]
            self.update_values(p,patch_q)
            self.update_regions(p)
            self.update_patches()
            num_iter+=1
        pass

    """Functions about output"""
    def display(self, test=0):
        if test:
            fig, axs = plt.subplots(1,4, figsize= (10,10))
            axs[0].imshow(self.image)
            axs[0].set_title("Original image"); axs[0].axis('off')
            axs[1].imshow(self.source_region)
            axs[1].set_title("Source Region"); axs[1].axis('off')
            axs[2].imshow(self.target_region)
            axs[2].set_title("Target Region"); axs[2].axis('off')
            axs[3].imshow(self.mask)
            axs[3].set_title("Mask"); axs[3].axis('off')
            plt.tight_layout()
        else:
            fig, axs = plt.subplots(figsize= (10,10))
            axs.imshow(self.source_region)
            axs.set_title("Source Region"); axs.axis('off')
        plt.show()

    def save_image(self, image_name):
        # At the end : output image = self.source_region
        plt.imsave("output/"+image_name, self.source_region)

#Pour tester:

if __name__ == "__main__":
    t0 = time.time()
    inpaint = Inpainting(patch_size=9, curr_im=len(original_filepaths)-3)
    inpaint.inpaint()
    delta_t=time.time()-t0
    min = delta_t//60
    sec = (((delta_t/60-min)*60)*100)//100
    print("Algorithm duration : "+str(min)+"mn"+str(sec)+"sec")
    inpaint.display(test=1)

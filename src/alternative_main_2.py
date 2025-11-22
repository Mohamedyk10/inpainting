import pandas as pd
import numpy as np
from glob import glob
from scipy import ndimage
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
from alternative_utils_2 import *
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
    def fast_load_image_from_database(self, filename_original, filename_mask):
        script_dir = os.path.dirname(__file__)
        data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))
        image_path = os.path.join(data_dir, filename_original)
        mask_path  = os.path.join(data_dir, filename_mask)

        self.image = plt.imread(image_path).copy()
        self.mask  = plt.imread(mask_path)

        if self.mask.ndim == 3:
            self.mask = self.mask[..., 0]
        self.mask = (self.mask > 0).astype(np.uint8)

    def get_contour(self):
        return self.target_region - ndimage.binary_erosion(self.target_region).astype(np.uint8)
    
    def get_source_region(self):
        return self.image * (1 - self.mask)[..., None] 
    
    def __init__(self, image_filename, mask_filename, patch_size=9, curr_im = 0):
        self.filename = image_filename
        self.current_img = curr_im
        self.image = np.array([])
        self.mask = np.array([])
        self.fast_load_image_from_database(image_filename, mask_filename)

        if self.image.dtype == np.uint8 or np.nanmax(self.image) > 1.0:
            self.image = self.image.astype(np.float32) / 255.0

        self.target_region = self.mask.copy()
        self.source_region = self.get_source_region()
        self.contour_region = self.get_contour()
        # Optimisation Set
        self.contour = {(int(x), int(y)) for x, y in np.argwhere(self.contour_region == 1)}

        self.source_patches = {}
        self.contour_patches = {}
        self.patch_size = patch_size
        
        # Optimisation Radius (CRUCIAL POUR LA VITESSE)
        img_min_dim = min(self.image.shape[0], self.image.shape[1])
        self.search_radius = int(img_min_dim * 0.20) # 20% de l'image par sécurité, ou remettre 0.10
        
        self.initialise_patch()

        self.confidence_values = {(i,j): int(self.target_region[i,j]==0) for i in range(len(self.target_region)) for j in range(len(self.target_region[1]))}
        self.priority_patches = {} 

        self.frames = [self.source_region.copy()]
        self.animation = None
        self.anim_fig = None
        # PAS DE PRE-CALCUL DE GRADIENT ICI !
    
    def calculate_priority(self):
        self.priority_patches = {} 
        for p in self.contour:
            confidence_term = calculate_confidence(p, self.confidence_values, self.patch_size)
            
            # Appel Dynamique : On passe l'image actuelle (self.source_region)
            data_term = calculate_dataterm(p, self.source_region, self.target_region) 
            
            self.priority_patches[p] = confidence_term * data_term
            
        # print(f"Priorités calculées pour {len(self.priority_patches)} patches.")

    def update_regions(self, p):
        x, y = p
        half = self.patch_size // 2
        i0, i1 = x - half, x + half + 1
        j0, j1 = y - half, y + half + 1

        points_filled = set()
        for i in range(i0, i1):
            for j in range(j0, j1):
                if 0 <= i < self.target_region.shape[0] and 0 <= j < self.target_region.shape[1]:
                    if self.target_region[i, j] == 1:
                        points_filled.add((i, j))

        self.contour.difference_update(points_filled)

        self.target_region[i0:i1, j0:j1] = 0
        self.source_region[i0:i1, j0:j1] = self.contour_patches[p]

        new_contour_points = set()
        h, w = self.target_region.shape
        i0_b = max(0, i0 - 1)
        i1_b = min(h, i1 + 1)
        j0_b = max(0, j0 - 1)
        j1_b = min(w, j1 + 1)
        
        for i in range(i0_b, i1_b):
            for j in range(j0_b, j1_b):
                if self.target_region[i, j] == 1:
                    if np.any(self.target_region[max(0, i-1):min(h, i+2), max(0, j-1):min(w, j+2)] == 0):
                        new_contour_points.add((i, j))
        
        self.contour.update(new_contour_points)
        return points_filled, new_contour_points
    
    def initialise_patch(self):
        half = self.patch_size//2
        self.source_patches = {(i,j): make_patch((i,j), self.source_region, self.patch_size) for i in range(half, self.image.shape[0]-half) for j in range(half, self.image.shape[1]-half) if np.all(self.target_region[i-half:i+half+1, j-half:j+half+1]==0 )}
        self.contour_patches = {(i, j): make_patch((i, j), self.source_region, self.patch_size) for (i, j) in self.contour if i-half >= 0 and i+half < self.image.shape[0] and j-half >= 0 and j+half < self.image.shape[1]}

    def update_patches(self, p, filled_points, new_points):
        half = self.patch_size // 2
        h, w = self.image.shape[:2]

        for point in filled_points:
            if point in self.contour_patches:
                del self.contour_patches[point]
        
        for point in new_points:
            if point[0] - half >= 0 and point[0] + half < h and point[1] - half >= 0 and point[1] + half < w:
                self.contour_patches[point] = make_patch(point, self.source_region, self.patch_size)

        for i in range(max(half, p[0]-self.patch_size*3//2), min(h-half, p[0]+self.patch_size*3//2+1)):
            for j in range(max(half, p[1]-self.patch_size*3//2), min(w-half, p[1]+self.patch_size*3//2+1)):
                if (i, j) not in self.source_patches:
                    target = make_patch((i,j), self.target_region, self.patch_size)
                    if np.all(target == 0):
                        self.source_patches[(i,j)] = make_patch((i,j), self.source_region, self.patch_size)

    def patch_to_use(self):
        if not self.priority_patches:
            if self.contour:
                 return random.choice(list(self.contour))
            raise IndexError("Contour vide.")
        return max(self.priority_patches, key=self.priority_patches.get)

    def best_match_sample(self, p):
        # Utilise le rayon de recherche pour la vitesse !
        return determine_closest_patch(self.target_region, self.source_patches, self.contour_patches, p, self.search_radius)

    def update_values(self,p,patch_q):
        half = self.patch_size // 2
        i0, i1 = p[0] - half, p[0] + half + 1
        j0, j1 = p[1] - half, p[1] + half + 1
        
        target_patch_mask = self.target_region[i0:i1, j0:j1]
        unknown_pixels_in_patch = (target_patch_mask == 1)
        
        new_patch_val = self.contour_patches[p].copy()
        new_patch_val[unknown_pixels_in_patch] = patch_q[unknown_pixels_in_patch]
        self.contour_patches[p] = new_patch_val
        
        priority = self.priority_patches[p]
        
        # Recalcul Dynamique pour la confiance
        data_term = calculate_dataterm(p, self.source_region, self.target_region) 

        if data_term > 0:
            confidence_to_propagate = priority / data_term 
        else:
            confidence_to_propagate = priority 
            
        for i in range(i0, i1):
            for j in range(j0, j1):
                if self.target_region[i, j] == 1: 
                    self.confidence_values[(i, j)] = confidence_to_propagate

    def inpaint(self, animate = True):
        if animate:
            self.generateAnimation()
        num_iter = 0 
        while np.any(self.target_region==1) and num_iter<3000:
            print("Iteration : " + str(num_iter))
            self.calculate_priority()
            p = self.patch_to_use()
            patch_p = self.contour_patches[p]
            q = self.best_match_sample(p); patch_q = self.source_patches[q]
            self.update_values(p,patch_q)
            filled_points, new_points = self.update_regions(p)
            self.update_patches(p, filled_points, new_points)
            num_iter+=1
            if animate:
                self.frames.append(self.source_region.copy())
        if animate:
            self.animate()

    def generateAnimation(self):
        fig, ax = plt.subplots(figsize=(10,10))
        im=ax.imshow(self.source_region)
        self.anim_fig = fig
        self.animation = im 
    def updateAnimation(self, frame):
        self.animation.set_array(self.frames[frame])
        return [self.animation]
    def animate(self):
        ani = FuncAnimation(self.anim_fig, self.updateAnimation, frames=len(self.frames), interval=50, blit=True)
        plt.show()
    def display(self, test=0, deb=0):
        # ... (votre code d'affichage inchangé) ...
        # Juste pour copier coller plus vite, je le mets ici:
        img8 = (self.image * 255).astype(np.uint8)
        if img8.shape[2] == 4: img8 = img8[..., :3]
        mask8 = (self.mask * 255).astype(np.uint8)
        if mask8.ndim == 3 and mask8.shape[2] == 1: mask8 = mask8.squeeze()
        if test:
            fig, axs = plt.subplots(1,4, figsize= (10,10))
            axs[0].imshow(self.image); axs[0].set_title("Original"); axs[0].axis('off')
            axs[1].imshow(self.source_region); axs[1].set_title("Our algo"); axs[1].axis('off')
            axs[2].imshow(self.mask); axs[2].set_title("Mask"); axs[2].axis('off')
            if deb: axs[3].imshow(self.target_region)
            else:
                t1 = time.time()
                inpainted = cv2.inpaint(img8, mask8, 3, cv2.INPAINT_TELEA)
                t2 = time.time()-t1
                axs[3].imshow(inpainted); axs[3].set_title("OpenCV"); axs[3].axis('off')
            plt.tight_layout()
        else:
            plt.imshow(self.source_region); plt.show()

    def save_image(self):
        image_name = get_image_name(self.filename)
        plt.imsave("output/"+image_name, self.source_region)

if __name__ == "__main__":
    t0 = time.time()
    # TEST SUR LE TRIANGLE (Simple)
    inpaint = Inpainting(image_filename='simple_triangle.png', mask_filename='simple-triangle.mask.webp', patch_size=12, curr_im=3)
    inpaint.inpaint()
    
    delta_t=time.time()-t0
    min = delta_t//60
    sec = round(delta_t % 60, 2)
    print(f"Algorithm duration : {min}mn {sec}sec")
    
    inpaint.display(test=1)
    answer = input("Save? (y/n)")
    if answer.lower()=="y":
        inpaint.save_image()
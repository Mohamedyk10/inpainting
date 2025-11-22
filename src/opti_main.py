import pandas as pd
import numpy as np
from glob import glob
from scipy import ndimage
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
from opti_utils import *
import os
import random
import time

# Rendre le chemin robuste
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

class Inpainting():
    def fast_load_image_from_database(self, filename_original, filename_mask, create_mask=False):
        script_dir = os.path.dirname(__file__)
        data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))
        image_path = os.path.join(data_dir, filename_original)

        self.image = plt.imread(image_path).copy()
        if create_mask:
            self.mask = add_mask_rect(image=self.image)
        else: 
            mask_path  = os.path.join(data_dir, filename_mask)
            self.mask  = plt.imread(mask_path)

        if self.mask.ndim == 3:
            self.mask = self.mask[..., 0]
        self.mask = (self.mask > 0).astype(np.uint8)

    def get_contour(self):
        return self.target_region - ndimage.binary_erosion(self.target_region).astype(np.uint8)
    
    def get_source_region(self):
        return self.image * (1 - self.mask)[..., None] 
    
    def __init__(self, image_filename, mask_filename, patch_size=9, search_prop=0.3, sigma_lissage=1.0, create_mask=False):
        self.filename = image_filename
        self.image = np.array([])
        self.mask = np.array([])
        self.fast_load_image_from_database(image_filename, mask_filename, create_mask)

        if self.image.dtype == np.uint8 or np.nanmax(self.image) > 1.0:
            self.image = self.image.astype(np.float32) / 255.0

        self.target_region = self.mask.copy()
        self.source_region = self.get_source_region()
        self.contour_region = self.get_contour()
        self.contour = {(int(x), int(y)) for x, y in np.argwhere(self.contour_region == 1)}

        self.source_patches = {}
        self.contour_patches = {}
        self.patch_size = patch_size
        self.sigma_lissage = sigma_lissage
        
        # Optimisation (Research radius + pré-calcul des gradients)
        img_min_dim = min(self.image.shape[0], self.image.shape[1])
        self.search_radius = int(img_min_dim * search_prop)
        
        self.initialise_patch()

        self.confidence_values = {(i,j): int(self.target_region[i,j]==0) for i in range(len(self.target_region)) for j in range(len(self.target_region[1]))}
        self.priority_patches = {} 

        print("Initialisation des gradients...")
        self.grads_y, self.grads_x = initialize_gradients(self.source_region, self.sigma_lissage)
        
        # On pré-calcule aussi le gradient du masque (qui change, mais on peut le garder ici ou le mettre à jour localement, 
        # pour simplifier on utilise np.gradient sur le masque dans calculate_priority, c'est rapide car le masque est simple)
        # Mais pour être cohérent avec calculate_dataterm_optimized, on a besoin des grads du masque
        # On va les calculer à la volée dans calculate_priority ou on peut les stocker.
        # Pour l'instant, calculons les dans calculate_priority car le masque change de forme binaire brute.
        
        # Attributs pour l'animation
        self.frames = [self.source_region.copy()]
        self.animation = None
        self.anim_fig = None
    
    def calculate_priority(self):
        self.priority_patches = {} 
        
        # Calcul du gradient du masque (Rapide sur image binaire)
        grad_mask_y, grad_mask_x = np.gradient(self.target_region.astype(np.float32))

        for p in self.contour:
            confidence_term = calculate_confidence(p, self.confidence_values, self.patch_size)
            
            # Appel à la fonction OPTIMISÉE (lecture seule)
            data_term = calculate_dataterm_optimized(
                p, 
                grad_mask_y, grad_mask_x, 
                
                self.grads_y, self.grads_x
            )
            
            self.priority_patches[p] = confidence_term * data_term

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

        # Optimisation: Mise à jour locale des source_patches
        for i in range(max(half, p[0]-self.patch_size*3//2), min(h-half, p[0]+self.patch_size*3//2+1)):
            for j in range(max(half, p[1]-self.patch_size*3//2), min(w-half, p[1]+self.patch_size*3//2+1)):
                if (i, j) not in self.source_patches:
                    target = make_patch((i,j), self.target_region, self.patch_size)
                    if np.all(target == 0):
                        self.source_patches[(i,j)] = make_patch((i,j), self.source_region, self.patch_size)

    def patch_to_use(self):
        if not self.priority_patches:
            if self.contour: return random.choice(list(self.contour))
            raise IndexError("Contour vide.")
        return max(self.priority_patches, key=self.priority_patches.get)

    def best_match_sample(self, p):
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
        
        # On a besoin du gradient du masque ici aussi
        grad_mask_y, grad_mask_x = np.gradient(self.target_region.astype(np.float32))
        
        # Recalcul rapide D(p) avec lecture dans les tableaux
        data_term = calculate_dataterm_optimized(
            p, 
            grad_mask_y, grad_mask_x, 
            self.grads_y, self.grads_x
        )

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
        while np.any(self.target_region==1):
            print("Iteration : " + str(num_iter))
            self.calculate_priority()
            p = self.patch_to_use()
            
            # Sauvegarder l'état
            q = self.best_match_sample(p); patch_q = self.source_patches[q]
            self.update_values(p,patch_q)
            filled_points, new_points = self.update_regions(p)
            
            # --- MISE A JOUR LOCALE DES GRADIENTS ---
            # On met à jour les tableaux self.grads_y et self.grads_x juste autour de p
            update_local_gradients(self.source_region, self.grads_y, self.grads_x, p, self.patch_size, self.sigma_lissage)
            # ----------------------------------------
            
            self.update_patches(p, filled_points, new_points)
            num_iter+=1
            if animate:
                self.frames.append((self.source_region*255).astype(np.uint8).copy())
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
        if 20*len(self.frames)>= 5000:
            interval = 5000//len(self.frames) # l'animation durera toujours 5 secondes
        else: interval = 20
        ani = FuncAnimation(self.anim_fig, self.updateAnimation, frames=len(self.frames), interval=interval, blit=True)
        plt.show()
    def display(self, test=0, deb=0):
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
            plt.imshow(self.source_region)
        plt.show()

    def save_image(self):
        image_name = get_image_name(self.filename)
        plt.imsave("output/"+image_name, self.source_region)

if __name__ == "__main__":
    """Ce sont les paramètres à ajuster pour l'inpainting"""
    create_mask = False # if True, allow to manually create a rectangular mask, the mask filename will be ignored
    patch_size = 7
    search_prop = 0.35
    sigma_lissage = 1.25
    t0 = time.time()
    #inpaint = Inpainting(image_filename='dog_example.png', mask_filename='dog_example.mask.webp', patch_size=patch_size, search_prop=search_prop, sigma_lissage=sigma_lissage, create_mask=create_mask)
    inpaint = Inpainting(image_filename='simple_triangle.png', mask_filename='simple-triangle.mask.webp', patch_size=9, search_prop=search_prop, sigma_lissage=sigma_lissage, create_mask=create_mask)
    inpaint.inpaint()
    
    delta_t=time.time()-t0
    min = delta_t//60
    sec = round(delta_t % 60, 2)
    print(f"Algorithm duration : {min}mn {sec}sec")
    
    inpaint.display(test=1)
    answer = input("Save? (y/n)")
    if answer.lower()=="y":
        inpaint.save_image()
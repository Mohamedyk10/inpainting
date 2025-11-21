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
    """A class that implements Exemplar-based inpainting method"""

    """Loading data"""
    def fast_load_image_from_database(self, filename_original="entete-textures.jpg", filename_mask="entete-textures.mask.webp"):
        # Chemin absolu du dossier 'data' (voisin de 'src')
        script_dir = os.path.dirname(__file__)
        data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))

        # Construit les chemins complets
        image_path = os.path.join(data_dir, filename_original)
        mask_path  = os.path.join(data_dir, filename_mask)

        # Chargement
        self.image = plt.imread(image_path).copy()  # ✅ .copy() pour rendre modifiable
        self.mask  = plt.imread(mask_path)

        # Si le masque est RGB → on garde un canal
        if self.mask.ndim == 3:
            self.mask = self.mask[..., 0]

        # Binarisation
        self.mask = (self.mask > 0).astype(np.uint8)

        print(f"Image chargée : {image_path}")
        print(f"Masque chargé : {mask_path}")
    
    # def load_image_from_database(self):
    #     self.image = plt.imread(original_filepaths[self.current_img])
    #     self.mask = plt.imread(mask_filepaths[self.current_img])

    #     #Si le masque est RGB 
    #     if self.mask.ndim == 3:
    #         self.mask = self.mask[...,0]

    #     #Pour le rendre binaire
    #     self.mask = (self.mask > 0).astype(np.uint8)
        
    #     if self.current_img<len(original_filepaths)-1:
    #         self.current_img+=1
    
    """Getters"""
    def get_contour(self):
        return self.target_region - ndimage.binary_erosion(self.target_region).astype(np.uint8)
    def get_source_region(self):
        return self.image * (1 - self.mask)[..., None] 
    def get_source_region(self):
        return self.image * (1 - self.mask)[..., None] 
    
    def __init__(self, image_filename, mask_filename, patch_size=9, curr_im = 0):
        self.filename = image_filename
        # Dataset related variables
        self.current_img=curr_im
        self.image = np.array([])
        self.mask = np.array([])
        self.fast_load_image_from_database(image_filename, mask_filename)

        if self.image.dtype == np.uint8 or np.nanmax(self.image) > 1.0:
            self.image = self.image.astype(np.float32) / 255.0

        # Regions 
        self.target_region = self.mask.copy()
        self.source_region = self.get_source_region()
        self.contour_region = self.get_contour()
        # Contour elements (array) -> Change from list [] to set {}
        self.contour = {(int(x), int(y)) for x, y in np.argwhere(self.contour_region == 1)}

        # Patches initialization
        self.source_patches = {}
        self.contour_patches={} # {(i,j): patch}
        self.patch_size = 9
        img_min_dim = min(self.image.shape[0], self.image.shape[1])
        self.search_radius = int(img_min_dim * 0.10)
        self.initialise_patch()

        # values for patches
        self.confidence_values = {(i,j): int(self.target_region[i,j]==0) for i in range(len(self.target_region)) for j in range(len(self.target_region[1]))} #{(i,j) : priority}
        self.data_terms = {} #same
        self.priority_patches = {} #same

        # Animation
        self.frames = [self.source_region.copy()]
        self.animation= None
        self.anim_fig = None

        print("Pré-calcul des gradients...")
        
        # 1. Pré-calcul du lissage et des gradients de l'image
        sigma_lissage = 1.0
        smoothed_region = ndimage.gaussian_filter(self.source_region, sigma=(sigma_lissage, sigma_lissage, 0))
        
        self.precomputed_gradients = []
        for channel in range(smoothed_region.shape[2]):
            I_channel = smoothed_region[:, :, channel]
            grad_y, grad_x = np.gradient(I_channel)
            self.precomputed_gradients.append((grad_y, grad_x))

        # 2. Pré-calcul du gradient du masque
        self.grad_mask_y, self.grad_mask_x = np.gradient(self.target_region.astype(np.float32))
        
        print("Pré-calcul terminé.")
    
    def calculate_priority(self):
        """Calcule la priorité P(p) = C(p) * D(p) pour chaque patch de contour."""
        
        self.priority_patches = {} 
        
        for p in self.contour:

            confidence_term = calculate_confidence(p, self.confidence_values, self.patch_size)
            
            data_term = calculate_dataterm(p,self.grad_mask_y,self.grad_mask_x,self.precomputed_gradients)
            
            self.priority_patches[p] = confidence_term * data_term
            
        print(f"Priorités calculées pour {len(self.priority_patches)} patches de contour.")

    def update_regions(self, p):
        x, y = p
        half = self.patch_size // 2
        i0, i1 = x - half, x + half + 1
        j0, j1 = y - half, y + half + 1

        # 1. Find points that will be filled
        points_filled = set()
        for i in range(i0, i1):
            for j in range(j0, j1):
                if 0 <= i < self.target_region.shape[0] and 0 <= j < self.target_region.shape[1]:
                    # Check if this pixel was part of the hole (and thus the contour)
                    if self.target_region[i, j] == 1:
                        points_filled.add((i, j))

        # 2. Remove filled points from the main contour set
        self.contour.difference_update(points_filled)

        # 3. Update the image and mask
        self.target_region[i0:i1, j0:j1] = 0
        self.source_region[i0:i1, j0:j1] = self.contour_patches[p]

        # 4. Find NEW contour points locally
        new_contour_points = set()
        h, w = self.target_region.shape
        i0_b = max(0, i0 - 1)
        i1_b = min(h, i1 + 1)
        j0_b = max(0, j0 - 1)
        j1_b = min(w, j1 + 1)
        
        for i in range(i0_b, i1_b):
            for j in range(j0_b, j1_b):
                # If this pixel IS in the hole (target_region == 1)...
                if self.target_region[i, j] == 1:
                    # ...and it has at least one *known* neighbor...
                    if np.any(self.target_region[max(0, i-1):min(h, i+2), max(0, j-1):min(w, j+2)] == 0):
                        # ...it's a new contour point. Add it.
                        new_contour_points.add((i, j))
        
        # 5. Add the new points to the main contour set
        self.contour.update(new_contour_points)
        
        # 6. Return the sets of changed points for update_patches
        return points_filled, new_contour_points
    
    def initialise_patch(self):
        half = self.patch_size//2
        self.source_patches = {(i,j): make_patch((i,j), self.source_region, self.patch_size) for i in range(half, self.image.shape[0]-half) for j in range(half, self.image.shape[1]-half) if np.all(self.target_region[i-half:i+half+1, j-half:j+half+1]==0 )}
        self.contour_patches = {(i, j): make_patch((i, j), self.source_region, self.patch_size) for (i, j) in self.contour if i-half >= 0 and i+half < self.image.shape[0] and j-half >= 0 and j+half < self.image.shape[1]}

    def update_patches(self, p, filled_points, new_points):
        """
        Incrementally update contour_patches and check for new source_patches locally.
        """
        half = self.patch_size // 2
        h, w = self.image.shape[:2]

        # 1. Remove patches for points that were just filled
        for point in filled_points:
            if point in self.contour_patches:
                del self.contour_patches[point]
        
        # 2. Add patches for new contour points
        for point in new_points:
            i, j = point
            # Ensure the patch is fully within bounds before creating it
            if i - half >= 0 and i + half < h and j - half >= 0 and j + half < w:
                self.contour_patches[point] = make_patch(point, self.source_region, self.patch_size)

        # 3. Check for new source_patches locally
        # (This loop checks a 3x3 patch-sized area around p)
        # This part is necessary to find new patches that can be used for matching.
        for i in range(max(half, p[0]-self.patch_size*3//2), min(h-half, p[0]+self.patch_size*3//2+1)):
            for j in range(max(half, p[1]-self.patch_size*3//2), min(w-half, p[1]+self.patch_size*3//2+1)):
                if (i, j) not in self.source_patches: # Only check if not already a source
                    target = make_patch((i,j), self.target_region, self.patch_size)
                    if np.all(target == 0):
                        self.source_patches[(i,j)] = make_patch((i,j), self.source_region, self.patch_size)

    def patch_to_use(self):
        """Retourne la clé du patch avec la priorité P(p) la plus élevée (max(C(p)))."""
    
        if not self.priority_patches:
        # Cas de secours si le contour est vide (ne devrait pas arriver si le trou existe)
            if self.contour:
                 return random.choice(list(self.contour))
            raise IndexError("Contour vide, l'inpainting est terminé ou un problème est survenu.")
        
    # Sélectionne la clé (coordonnée) ayant la valeur (priorité) maximale
        p_max = max(self.priority_patches, key=self.priority_patches.get)
        return p_max

    def best_match_sample(self, p): # p = (i,j)
        """Returns the best match patch"""
        return determine_closest_patch(self.target_region, self.source_patches, self.contour_patches, p, self.search_radius)

    def update_values(self,p,patch_q):
        """Met à jour les valeurs de pixel du patch (p) et la confiance des pixels remplis."""
    
        half = self.patch_size // 2
        i0, i1 = p[0] - half, p[0] + half + 1
        j0, j1 = p[1] - half, p[1] + half + 1
        
        # 1. Mise à jour des valeurs de pixel du patch p
        target_patch_mask = self.target_region[i0:i1, j0:j1]
        unknown_pixels_in_patch = (target_patch_mask == 1)
        
        new_patch_val = self.contour_patches[p].copy()
        new_patch_val[unknown_pixels_in_patch] = patch_q[unknown_pixels_in_patch]
        self.contour_patches[p] = new_patch_val
        
        # 2. MISE À JOUR DE LA CONFIANCE
 
        priority = self.priority_patches[p]
        
        # Recalcul de D(p)
        data_term = calculate_dataterm( p, self.grad_mask_y, self.grad_mask_x, self.precomputed_gradients )

        if data_term > 0:

            confidence_to_propagate = priority / data_term 
        else:
        
            confidence_to_propagate = priority 
            
        # Mettre à jour la confiance de chaque pixel qui vient d'être rempli
        for i in range(i0, i1):
            for j in range(j0, j1):
                if self.target_region[i, j] == 1: 
                    self.confidence_values[(i, j)] = confidence_to_propagate


    """Main Function"""
    def inpaint(self, animate = True):
        if animate:
            self.generateAnimation()
        num_iter = 0 # Juste pour le débuggage, à retirer ensuite
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
        pass
    
    """Functions about output"""
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
        img8 = (self.image * 255).astype(np.uint8)

        if img8.shape[2] == 4:
            img8 = img8[..., :3]

        mask8 = (self.mask * 255).astype(np.uint8)

        if mask8.ndim == 3 and mask8.shape[2] == 1:
            mask8 = mask8.squeeze()

        if test:
            fig, axs = plt.subplots(1,4, figsize= (10,10))
            axs[0].imshow(self.image)
            axs[0].set_title("Original image"); axs[0].axis('off')
            axs[1].imshow(self.source_region)
            axs[1].set_title("Our algorithm"); axs[1].axis('off')
            if deb:
                axs[2].imshow(self.target_region)
                axs[2].set_title("Target Region"); axs[2].axis('off')
            else:
                # Inpainting algorithm for opencv
                '''img8 = (np.clip(self.image, 0, 1) * 255).astype(np.uint8)
                mask8 = (self.mask.astype(np.uint8) * 255)''' # NE PAS RE-CONVERTIR ! Sinn des fois il y a erreur de channel
                t1 = time.time()
                inpainted = cv2.inpaint(img8, mask8, 3, cv2.INPAINT_TELEA)
                t2 = time.time()-t1
                axs[2].imshow(inpainted)
                axs[2].set_title("Inpaint (OpenCV)"); axs[2].axis('off')
                print(f"Opencv2 inpainting duration : {t2//60}mn{t2/60-t2//60}sec")
            axs[3].imshow(self.mask)
            axs[3].set_title("Mask"); axs[3].axis('off')
            plt.tight_layout()
        else:
            fig, axs = plt.subplots(figsize= (10,10))
            axs.imshow(self.source_region)
            axs.set_title("Source Region"); axs.axis('off')
        plt.show()

    def save_image(self):
        # At the end : output image = self.source_region
        image_name = get_image_name(self.filename)
        plt.imsave("output/"+image_name, self.source_region)

#Pour tester:

if __name__ == "__main__":
    t0 = time.time()
    inpaint = Inpainting(image_filename='entete-textures.jpg', mask_filename='entete-textures.mask.webp', patch_size=9, curr_im=3)
    inpaint.inpaint()
    delta_t=time.time()-t0
    min = delta_t//60
    sec = (((delta_t/60-min)*60)*100)//100
    print("Algorithm duration : "+str(min)+"mn"+str(sec)+"sec")
    inpaint.display(test=1)
    answer = input("Would you like to save the image ? (y/n)")
    if answer.lower()=="y":
        inpaint.save_image()


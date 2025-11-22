import pandas as pd
import numpy as np
from glob import glob
from scipy import ndimage
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
from alternative_utils import *
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
    """A class that implements Exemplar-based inpainting method based on the article."""

    """Loading data"""
    def fast_load_image_from_database(self, filename_original="entete-textures.jpg", filename_mask="entete-textures.mask.webp", create_mask=0):
        """Loads our image and mask
        - If ``create_mask`` is true, allow us to generate a mask by inputing the 4 values defining the rectangle"""
        # Chemin absolu du dossier 'data' (voisin de 'src')
        script_dir = os.path.dirname(__file__)
        data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))

        # Chargement de l'image
        image_path = os.path.join(data_dir, filename_original)
        self.image = plt.imread(image_path).copy()  # ✅ .copy() pour rendre modifiable

        # Chargement du masque
        if create_mask:
            self.mask = add_mask_rect(image=self.image)
        else:
            mask_path  = os.path.join(data_dir, filename_mask)
            self.mask  = plt.imread(mask_path)

        # Si le masque est RGB → on garde un canal
        if self.mask.ndim == 3:
            self.mask = self.mask[..., 0]

        # Binarisation
        self.mask = (self.mask > 0).astype(np.uint8)

        print(f"Image chargée : {filename_original}")
        print(f"Masque chargé")
    
    """Getters and initialisation"""
    def get_contour(self):
        return self.target_region - ndimage.binary_erosion(self.target_region).astype(np.uint8)
    def get_source_region(self):
        return self.image * (1 - self.mask)[..., None] 
    
    def __init__(self, image_filename, mask_filename, patch_size=9, create_mask=0):
        """Initialize our inpainting algorithm.
        - ``image_filename`` the name of the image file to inpaint.
        - ``mask_filename`` the name of the mask file to inpaint (if **create_mask=1**, it is not used).
        - ``patch_size`` the size of patches created by the algorithm.
        - ``create_mask``, if True, allow us to create rectangular masks by giving the 4 values that defines a rectangle :
            - x1 : left column
            - y1 : higher row
            - x2 : right column
            - y2 : lower row

            Note that in image processing : if y>y', it means that y is below y'.
            """
        self.filename = image_filename
        # Dataset related variables
        self.image = np.array([])
        self.mask = np.array([])
        self.fast_load_image_from_database(image_filename, mask_filename, create_mask)

        if self.image.dtype == np.uint8 or np.nanmax(self.image) > 1.0:
            self.image = self.image.astype(np.float32) / 255.0

        # Regions 
        self.target_region = self.mask.copy()
        self.source_region = self.get_source_region()
        self.contour_region = self.get_contour()
        # Contour elements (array)
        self.contour = {(int(x), int(y)) for x, y in np.argwhere(self.contour_region == 1)}

        # Patches initialization
        self.source_patches = {}
        self.contour_patches={} # {(i,j): patch}
        self.patch_size = patch_size
        self.initialise_patch()

        # values for patches
        self.confidence_values = {(i,j): int(self.target_region[i,j]==0) for i in range(len(self.target_region)) for j in range(len(self.target_region[1]))} #{(i,j) : priority}
        self.data_terms = {} #same
        self.priority_patches = {} #same

        # Animation
        self.frames = [self.source_region.copy()]
        self.animation= None
        self.anim_fig = None
    
    def initialise_patch(self):
        half = self.patch_size//2
        self.source_patches = {(i,j): make_patch((i,j), self.source_region, self.patch_size) for i in range(half, self.image.shape[0]-half) for j in range(half, self.image.shape[1]-half) if np.all(self.target_region[i-half:i+half+1, j-half:j+half+1]==0 )}
        self.contour_patches = {(i, j): make_patch((i, j), self.source_region, self.patch_size) for (i, j) in self.contour if i-half >= 0 and i+half < self.image.shape[0] and j-half >= 0 and j+half < self.image.shape[1]}
    
    """Priority based functions"""
    def calculate_priority(self, convertBW=True):
        """Calculate priority P(p)=C(p).D(p) for each patch of the contour."""
        
        self.priority_patches = {} 
        for p in self.contour:
            confidence_term = calculate_confidence(p, self.confidence_values, self.patch_size)
            if convertBW:
                data_term = calculate_dataterm2(p, self.source_region, self.target_region) 
            else:
                data_term = calculate_dataterm(p, self.source_region, self.target_region)
            
            self.priority_patches[p] = confidence_term * data_term
            
        print(f"Priorités calculées pour {len(self.priority_patches)} patches de contour.")
    def patch_to_use(self):
        """Retourne la clé du patch avec la priorité P(p) la plus élevée (max(C(p)))."""
    
        if not self.priority_patches or all(v == 0 for v in self.priority_patches.values()):
        # Cas de secours si le contour est vide (ne devrait pas arriver si le trou existe)
            if self.contour:
                 return random.choice(list(self.contour))
            raise IndexError("Contour vide, l'inpainting est terminé ou un problème est survenu.")
        
    # Sélectionne la clé (coordonnée) ayant la valeur (priorité) maximale
        p_max = max(self.priority_patches, key=self.priority_patches.get)
        return p_max

    def best_match_sample(self, p): # p = (i,j)
        """Returns the best match patch"""
        return determine_closest_patch(self.target_region, self.source_patches, self.contour_patches, p)

    """Updating"""
    def update_regions(self, p):
        x,y = p
        i0, i1 = x-self.patch_size//2, x+self.patch_size//2+1
        j0, j1 = y-self.patch_size//2, y+self.patch_size//2+1
        self.target_region[i0:i1, j0:j1]=0
        self.source_region[i0:i1, j0:j1] = self.contour_patches[p]
        self.contour_region = self.get_contour()
        self.contour = [(int(x), int(y)) for x, y in np.argwhere(self.contour_region == 1)]
        pass
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
    def update_patches(self, p, filled_points, new_points):
        """Create a patch for a pixel in the contour"""
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
        for i in range(max(half,p[0]-self.patch_size*3//2), min(len(self.image)-half,p[0]+self.patch_size*3//2+1)):
            for j in range(max(half,p[1]-self.patch_size*3//2), min(len(self.image[0])-half,p[1]+self.patch_size*3//2+1)):
                target = make_patch((i,j), self.target_region, self.patch_size)
                if np.all(target==0):
                    self.source_patches[(i,j)]=make_patch((i,j), self.source_region, self.patch_size)


    def update_values(self,p,patch_q, convertBW=True):
        """Update values of unkown pixels p and their confidence values"""
    
        half = self.patch_size // 2
        i0, i1 = p[0] - half, p[0] + half + 1
        j0, j1 = p[1] - half, p[1] + half + 1
        
        # Updating values of unkown pixels p
        target_patch_mask = self.target_region[i0:i1, j0:j1]
        unknown_pixels_in_patch = (target_patch_mask == 1)
        
        new_patch_val = self.contour_patches[p].copy()
        new_patch_val[unknown_pixels_in_patch] = patch_q[unknown_pixels_in_patch]
        self.contour_patches[p] = new_patch_val
        
        # 2. Updating confidence values
 
        priority = self.priority_patches[p]
        
        if convertBW:
            data_term =calculate_dataterm2(p, self.source_region, self.target_region) 
        else:
            data_term =calculate_dataterm(p, self.source_region, self.target_region)

        if data_term > 0:
            confidence_to_propagate = priority / data_term 
        else:
            confidence_to_propagate = priority 
            
        for i in range(i0, i1):
            for j in range(j0, j1):
                if self.target_region[i, j] == 1: 
                    self.confidence_values[(i, j)] = confidence_to_propagate


    """Main Function"""
    def inpaint(self, animate = True, convertBW=True):
        """runs the exemplar-based inpainting algorithm

            - **animate** = True : allows us to see the animation of the filling process

            - **convertBW** select the calculating dataterm method.

                - If convertBW=True : We convert the image in BW before calculating gradients.

                - Otherwise : We use the three RGB canals and select the maximum isotope. (way slower and doesn't give too much better results than the first) 
                """
        if animate:
            self.generateAnimation()
        num_iter = 0 # Juste pour le débuggage, à retirer ensuite
        while np.any(self.target_region==1) and num_iter<3000:
            print("Iteration : " + str(num_iter))
            t0 = time.time()
            self.calculate_priority(convertBW=convertBW)
            t1 = time.time();print(f"Calculate Priority : {t1-t0}")
            p = self.patch_to_use()
            t0 = time.time();print(f"Patch to use : {-(t1-t0)}")
            q = self.best_match_sample(p); patch_q = self.source_patches[q]
            t1 = time.time();print(f"Best match sample : {t1-t0}")
            self.update_values(p,patch_q, convertBW=convertBW)
            t0 = time.time();print(f"Update values: {-(t1-t0)}")
            filled_points, new_points = self.update_regions(p)
            t1 = time.time();print(f"Update regions : {t1-t0}")
            self.update_patches(p, filled_points, new_points)
            t0 = time.time();print(f"Update patches: {-(t1-t0)}")
            num_iter+=1
            if num_iter%20 == 0:
                print(f"Le nombre de pixels inconnus : {len(self.target_region[self.target_region==1])}")
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
        self.anim_fig.gca().axis('off')
        self.anim_fig.suptitle("Step-by-Step visualisation", fontsize=16)
        plt.show()

    def display(self, test=0, deb=0):
        """Displays the inpainted image with some other things :
        - If ``test=0`` : shows the inpainted image only.
        - If ``test=1`` :
            - If ``deb=0``, shows :
                - The original image
                - The inpainted image
                - The mask used
                - The Target Region (should be empty : a null image)

                Used to verify that the algorithm has converged.
            - If ``deb=1``, shows :
                - The original image
                - The inpainted image
                - The mask used
                - The inpainted image with OpenCV.
            """
        img8 = (self.image * 255).astype(np.uint8)

        if img8.shape[2] == 4:
            img8 = img8[..., :3]

        mask8 = (self.mask * 255).astype(np.uint8)

        if mask8.ndim == 3 and mask8.shape[2] == 1:
            mask8 = mask8.squeeze()

        if test:
            """Used to see the result of our algorithm"""
            fig, axs = plt.subplots(1,4, figsize= (10,10))
            axs[0].imshow(self.image)
            axs[0].set_title("Original image"); axs[0].axis('off')
            axs[1].imshow(self.source_region)
            axs[1].set_title("Our algorithm"); axs[1].axis('off')
            axs[2].imshow(self.mask)
            axs[2].set_title("Mask"); axs[2].axis('off')
            if deb:
                """Was used in the begining to verify that the algorithm 
                has really filled up the whole area of the mask"""
                axs[3].imshow(self.target_region)
                axs[3].set_title("Target Region"); axs[3].axis('off')
            else:
                # Inpainting algorithm for opencv
                '''img8 = (np.clip(self.image, 0, 1) * 255).astype(np.uint8)
                mask8 = (self.mask.astype(np.uint8) * 255)''' # NE PAS RE-CONVERTIR ! Sinn des fois il y a erreur de channel
                t1 = time.time()
                inpainted = cv2.inpaint(img8, mask8, 3, cv2.INPAINT_TELEA)
                t2 = time.time()-t1
                axs[3].imshow(inpainted)
                axs[3].set_title("Inpaint (OpenCV)"); axs[3].axis('off')
            plt.tight_layout()
        else:
            """Plots only the result of our algorithm"""
            fig, axs = plt.subplots(figsize= (10,10))
            axs.imshow(self.source_region)
            axs.set_title("Source Region"); axs.axis('off')
        plt.show()

    def save_image(self):
        """Saves the inpainted image in the ``output/`` folder"""
        # At the end : output image = self.source_region
        image_name = get_image_name(self.filename)
        plt.imsave("output/"+image_name, self.source_region)

if __name__ == "__main__":
    t0 = time.time()
    """Some suggestions of mask shapes if you want to create ones :
        8.original.webp : 172, 241, 239, 365
        entete-textures : 432, 23, 600, 100
        simple_triangle : 30, 110, 77, 158
    In order to make personal rectangular masks you need to make sure that create_mask = 1 
    """
    #inpaint = Inpainting(image_filename='8.original.webp', mask_filename='8.mask.webp', patch_size=25, create_mask=0)
    #inpaint = Inpainting(image_filename='simple_triangle.png', mask_filename='simple-triangle.mask.webp', patch_size=6, create_mask=0)
    #inpaint = Inpainting(image_filename='entete-textures.jpg', mask_filename='entete-textures.mask.webp', patch_size=6, create_mask=1)
    inpaint = Inpainting(image_filename='dog_example.png', mask_filename='dog_example.mask.webp', patch_size=9, create_mask=0)
    inpaint.inpaint()
    delta_t=time.time()-t0
    min = delta_t//60
    sec = (((delta_t/60-min)*60)*100)//100
    print("Algorithm duration : "+str(min)+"mn"+str(sec)+"sec")
    inpaint.display(test=1)
    answer = input("Would you like to save the image ? (y/n)")
    if answer.lower()=="y":
        inpaint.save_image()


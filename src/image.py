import pandas as pd
import numpy as np

import cv2
import matplotlib.pylab as plt

"""Il va falloir parcourir les images de data/.
Une bibliothèque intéressante s'appelle glob, qui permet de
facilement parcourir les images et range les filepath dans une liste"""
filepath = [''] 

img_cv2 = cv2.imread(filepath)
img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

def load_images_from_database():
    pass

def display_image(image):
    fig, ax = plt.subplots(figsize= (10,10))
    ax.imshow(image)
    ax.axis('off')
    plt.show()

def save_image(image_name, image):
    a = cv2.imwrite("output/"+image_name, image)
    if not(a):
        print("Error during writing.")



import pandas as pd
import numpy as np

from glob import glob

import matplotlib.pylab as plt

from inpainting import *

original_filepaths = glob("../data/*.original.webp")
mask_filepaths = glob("../data/*.mask.webp")
current_img=0

image = None
mask = None

def load_image_from_database():
    image = plt.imread(original_filepaths[current_img])
    mask = plt.imread(mask_filepaths[current_img])
    current_img+=1

def display_image(image):
    fig, ax = plt.subplots(figsize= (10,10))
    ax.imshow(image)
    ax.axis('off')
    plt.show()

def save_image(image_name, image):
    plt.imsave("output/"+image_name, image)


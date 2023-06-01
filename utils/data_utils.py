import numpy as np
import pandas as pd
import os, re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import xml.etree.ElementTree as ET

from skimage.transform import resize
from tqdm import tqdm

# GLOBAL VARIABLES
# CROHME PATH
CROHME_PATH = os.path.abspath(__file__)
CROHME_PATH = re.findall('(.+/ImageToLatex).*', CROHME_PATH)[0]
CROHME_PATH = os.path.join(CROHME_PATH, 'ImageToLatex/data/CROHME')

# DATASET LOCATIONS
CROHME_TRAIN = os.path.join(CROHME_PATH, 'train')
CROHME_VAL = os.path.join(CROHME_PATH, 'val')
IMG_SIZE = (128, 128)

# GENERATE IMAGES
def generate_image(INKML_file, img_size=IMG_SIZE):
    '''
    :param INKML_file: contains the stroke data and the annotation truth. The stroke data is stored as trace element
    :param img_size: size of the output image
    :return: the image of the equation
    '''
    # Parse the XML file
    tree = ET.parse(INKML_file)
    root = tree.getroot()
    # Extract the traced points
    traces = []
    for trace in root.findall('{http://www.w3.org/2003/InkML}trace'):
        points = trace.text.strip().split(',')
        traces.append([(float(pt.split()[0]), float(pt.split()[1])) for pt in points])

    # Plot the points onto an image
    plt.figure(figsize=(6, 6))
    for trace in traces:
        plt.plot(*zip(*trace), c='black')
    plt.gca().invert_yaxis()  # INKML's origin is at the bottom-left
    plt.axis('off')  # Don't display axes
    plt.show()

generate_image("/home/mathador/Documents/ImageToLatex/data/CROHME/train/INKML/001-equation000.inkml")
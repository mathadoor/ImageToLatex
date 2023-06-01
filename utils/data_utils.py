import numpy as np
from skimage.draw import line
from skimage.io import imsave
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
CROHME_PATH = os.path.join(CROHME_PATH, 'data/CROHME')

# DATASET LOCATIONS
CROHME_TRAIN = os.path.join(CROHME_PATH, 'train')
CROHME_VAL = os.path.join(CROHME_PATH, 'val')
IMG_SIZE = (128, 128)

# GENERATE IMAGES
def generate_image(INKML_file, img_loc, img_size=IMG_SIZE):
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

    # Get the image size
    dpi = 100
    fig_size = (img_size[0] / dpi, img_size[1] / dpi)

    # Create an empty image
    # Determine the bounds of the traces
    min_x = min(point[0] for trace in traces for point in trace)
    max_x = max(point[0] for trace in traces for point in trace)
    min_y = min(point[1] for trace in traces for point in trace)
    max_y = max(point[1] for trace in traces for point in trace)

    # Create an empty image
    img = 255 * np.ones(img_size, dtype=np.uint8)

    # Draw the traces onto the image
    for trace in traces:
        for i in range(len(trace) - 1):
            # Scale the trace coordinates to the image size
            x1 = int((trace[i][0] - min_x) / (max_x - min_x) * (img_size[1] - 1))
            y1 = int((trace[i][1] - min_y) / (max_y - min_y) * (img_size[0] - 1))
            x2 = int((trace[i + 1][0] - min_x) / (max_x - min_x) * (img_size[1] - 1))
            y2 = int((trace[i + 1][1] - min_y) / (max_y - min_y) * (img_size[0] - 1))

            rr, cc = line(y1, x1, y2, x2)
            img[rr, cc] = 0

    # Save the image
    imsave(img_loc + INKML_file.split('/')[-1].split('.')[0] + '.png', img)

generate_image(CROHME_TRAIN + "/INKML/001-equation000.inkml", CROHME_TRAIN + "/IMG_RENDERED/")
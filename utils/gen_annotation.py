import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import xml.etree.ElementTree as ET

from skimage.transform import resize
from tqdm import tqdm

# Set the directory where the input files are
crohme_path = './data/CROHME/'
datasets = ['train', 'val', 'synthetic_data']
dataset_file = 'offline.csv'

# Read the csv file, get the image name,
df = pd.read_csv(os.path.join(crohme_path, dataset_file))
df['img_name'] = df['img'].apply(lambda x: x.split('/')[-1])
df['inkml_name'] = df['img_name'].apply(lambda x: os.path.join(crohme_path, 'train/INKML', x.split('.')[0] + '.inkml'))

def get_annotation(row):
    """
    Get the annotation from the inkml file
    """
    inkml_file = row['inkml_name']
    tree = ET.parse(inkml_file)
    root = tree.getroot()
    annotation = []
    for child in root:
        if child.tag == 'annotation':
            annotation.append(child.text)
    return annotation

df['annotation'] = df.apply(get_annotation, axis=1)
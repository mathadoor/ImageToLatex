import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import xml.etree.ElementTree as ET
import cv2

from skimage.transform import resize
from tqdm import tqdm

# Set the directory where the input files are
path = 'data/ICFHR_package/CROHME2011_data/CROHME_training/CROHME_training'


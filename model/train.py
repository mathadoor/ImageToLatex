from utils.data_loader import ImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os, re

CROHME_PATH = os.path.abspath(__file__)
CROHME_PATH = re.findall('(.+/ImageToLatex).*', CROHME_PATH)[0]
CROHME_PATH = os.path.join(CROHME_PATH, 'data/CROHME')


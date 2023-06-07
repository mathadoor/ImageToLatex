from utils.datasets import ImageDataset
from utils.global_params import CROHME_TRAIN, CNN_INPUT_DIM, BASE_CONFIG
from torch.utils.data import DataLoader
from models import VanillaWAP
import torchvision.transforms as transforms
import pandas as pd

BATCH_SIZE = 32
train_data_csv = pd.read_csv(CROHME_TRAIN + '/train.csv')

# Define your transforms
transform = transforms.Compose([transforms.Resize(CNN_INPUT_DIM), transforms.ToTensor()])


dataset = ImageDataset(train_data_csv['image_loc'], train_data_csv['label'], transform=transform)

# Define your dataloader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = VanillaWAP(BASE_CONFIG)
# Iterate through DataLoader
for x, y in dataloader:
    print(model(x).shape)
    break
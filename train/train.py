from utils.datasets import ImageDataset, collate_fn
from utils.global_params import CNN_INPUT_DIM, BASE_CONFIG, TR_IMAGE_SIZE, BATCH_SIZE, CROHME_TRAIN, VOCAB_LOC
from torch.utils.data import DataLoader
from models import VanillaWAP
import torchvision.transforms as transforms
import pandas as pd


train_data_csv = pd.read_csv(CROHME_TRAIN + '/train.csv')

# Define your transforms
transform = transforms.Compose([transforms.Resize(TR_IMAGE_SIZE), transforms.ToTensor()])
dataset = ImageDataset(train_data_csv['image_loc'], train_data_csv['label'], VOCAB_LOC, transform=transform)

# Define your dataloader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Model
model = VanillaWAP(BASE_CONFIG)
# Iterate through DataLoader
for x, y in dataloader:
    print(model(x).shape, y.shape)
    break

print(BASE_CONFIG['output_dim'])
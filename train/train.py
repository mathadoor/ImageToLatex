from utils.datasets import ImageDataset, collate_fn
from utils.global_params import BASE_CONFIG, TR_IMAGE_SIZE, BATCH_SIZE, CROHME_TRAIN, VOCAB_LOC
from torch.utils.data import DataLoader
import torch
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
    # Get Maximum length of a sequence in the batch, and use it to trim the output of the model
    # y.shape is (B, MAX_LEN) and x.shape is (B, L ,V) which is to be trimmed
    max_len = y.shape[1]
    p = model(x)[:,:max_len,:] # (B, L, V)

    # One hot encode y
    y = torch.nn.functional.one_hot(y, model.config['vocab_size']).float()

    # Compute Loss as cross entropy
    loss = torch.nn.functional.cross_entropy(p, y)
    print(loss)
    break
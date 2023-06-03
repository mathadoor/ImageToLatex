from utils.datasets import ImageDataset
from utils.data_utils import get_path
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd

CROHME_TRAIN = get_path(kind = 'train')
print(CROHME_TRAIN)
BATCH_SIZE = 32
IMAGE_SIZE = 32
train_data_csv = pd.read_csv(CROHME_TRAIN + '/train.csv')

# Define your transforms
transform = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])


dataset = ImageDataset(train_data_csv['image_loc'], train_data_csv['label'], transform=transform)

# Define your dataloader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Iterate through DataLoader
for x, y in dataloader:
    print(x.shape)
    break
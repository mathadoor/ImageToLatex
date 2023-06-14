from utils.datasets import ImageDataset, collate_fn
from utils.global_params import BASE_CONFIG, TR_IMAGE_SIZE, BATCH_SIZE, CROHME_TRAIN, VOCAB_LOC
from torch.utils.data import DataLoader
import torch
from models import VanillaWAP
import torchvision.transforms as transforms
import pandas as pd

# Define Dataset
train_data_csv = pd.read_csv(CROHME_TRAIN + '/train.csv') # Location

# Define transforms
transform = transforms.Compose([transforms.Resize(TR_IMAGE_SIZE), transforms.ToTensor()])
dataset = ImageDataset(train_data_csv['image_loc'], train_data_csv['label'], VOCAB_LOC, transform=transform)

# Define dataloader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Model
model = VanillaWAP(BASE_CONFIG)

# Training Constructs
train_params = BASE_CONFIG['train_params']
optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=train_params['lr_decay_step'],
                                            gamma=train_params['lr_decay'])

# Iterate through DataLoader
i = 0
for x, y in dataloader:
    # Get Maximum length of a sequence in the batch, and use it to trim the output of the model
    # y.shape is (B, MAX_LEN) and x.shape is (B, L ,V) which is to be trimmed
    max_len = y.shape[1]
    p = model(x)[:,:max_len,:] # (B, L, V)

    # One hot encode y
    y = torch.nn.functional.one_hot(y, model.config['vocab_size']).float()

    # Compute Loss as cross entropy
    loss = torch.nn.functional.cross_entropy(p, y)

    # Backpropagate
    loss.backward()
    optimizer.step()
    scheduler.step()

    optimizer.zero_grad()

    # Print loss
    if i % train_params['print_every'] == 0:
        print(f'Loss at step {i}: {loss.item()}')

    i += 1
    # Save model
    # model.save(i)
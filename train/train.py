from utils.datasets import ImageDataset, collate_fn
from utils.global_params import BASE_CONFIG, TR_IMAGE_SIZE, BATCH_SIZE, CROHME_TRAIN, VOCAB_LOC
from torch.utils.data import DataLoader, random_split
from torcheval.metrics import WordErrorRate
import torch
from models import VanillaWAP
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm

# Define Dataset
train_data_csv = pd.read_csv(CROHME_TRAIN + '/train.csv') # Location

# Define transforms
transform = transforms.Compose([transforms.Resize(TR_IMAGE_SIZE), transforms.ToTensor()])
dataset = ImageDataset(train_data_csv['image_loc'], train_data_csv['label'], VOCAB_LOC, transform=transform)

# Model
model = VanillaWAP(BASE_CONFIG)

# Training Constructs
train_params = BASE_CONFIG['train_params']
optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=train_params['lr_decay_step'],
                                            gamma=train_params['lr_decay'])

# Evaluation Constructs
wer = WordErrorRate(device=BASE_CONFIG['DEVICE'])

# Define dataloader
genrator = torch.Generator().manual_seed(train_params['random_seed'])
train, val = random_split(dataset, [0.8, 0.2], generator=genrator)
dataloader_train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dataloader_val = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Define AverageMeter
class AverageMeter:

    def __init__(self):
        self.total = None
        self.count = None
        self.reset()

    def reset(self):
        self.count = 0
        self.total = 0

    def update(self, value):
        self.total += value
        self.count += 1

    def compute(self):
        return self.total / self.count

def convert_to_string(tensor, index_to_word):
    """
    :param tensor: (B, L)
    :param index_to_word: dict
    :return:
    """
    tensor = tensor.tolist()
    string = ''
    for i in tensor:
        if i in [0, 2, 3]:
            continue
        string += index_to_word[i] + ' '

    return string.strip()

# Setup Training Loop
train_loss, val_loss = AverageMeter(), AverageMeter()

for i in range(train_params['epochs']):
    print("Epoch: ", i)
    for x, y in tqdm(dataloader_train):
        # Get Maximum length of a sequence in the batch, and use it to trim the output of the model
        # y.shape is (B, MAX_LEN) and x.shape is (B, L ,V) which is to be trimmed
        max_len = y.shape[1]
        p = model(x)[:,:max_len,:] # (B, L, V)

        # One hot encode y
        y = torch.nn.functional.one_hot(y, model.config['vocab_size']).float()

        # Compute Loss as cross entropy
        loss = torch.nn.functional.cross_entropy(p, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        optimizer.zero_grad()

        # Update loss
        train_loss.update(loss.item())
    print(f'\tTraining Loss during epoch {i}: {train_loss.compute()}')

    with torch.no_grad():
        model.eval()
        for x, y in dataloader_val:
            # Set model to eval mode
            max_len = y.shape[1]
            p = model(x)

            # Compute Loss as cross entropy
            loss = torch.nn.functional.cross_entropy(p[:,:max_len,:], torch.nn.functional.one_hot(y, model.config['vocab_size']).float())
            val_loss.update(loss.item())

            # Computer WER
            y_pred = torch.argmax(p, dim=1).detach().cpu().numpy()
            y_pred = [convert_to_string(y_pred[i, :], dataset.index_to_word) for i in range(y_pred.shape[0])]
            y_true = y.detach().cpu().numpy()
            y_true = [convert_to_string(y_true[i, :], dataset.index_to_word) for i in range(y_true.shape[0])]
            wer.update(y_pred, y_true)
    print(f'\tValidation WER during epoch {i}: {wer.compute()}')
    print(f'\tValidation Loss during epoch {i}: {val_loss.compute()}')

    # Reset AverageMeters
    train_loss.reset()
    val_loss.reset()
    wer.reset()

    # Save model
    model.save(i)
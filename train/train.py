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
optimizer = torch.optim.AdamW(model.parameters(), lr=train_params['lr'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=train_params['lr_decay_step'],
                                            gamma=train_params['lr_decay'])
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

# Evaluation Constructs
wer = WordErrorRate(device=BASE_CONFIG['DEVICE'])

# Define dataloader
generator = torch.Generator().manual_seed(train_params['random_seed'])
train, val = random_split(dataset, [0.8, 0.2], generator=generator)
dataloader_train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dataloader_val = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Define AverageMeter
class AverageMeter:

    def __init__(self, best=False, best_type='min'):
        self.total = None
        self.count = None
        if best:
            self.best = None
            self.best_type = best_type
        self.reset()

    def reset(self):
        self.count = 0
        self.total = 0

    def update(self, value):
        self.total += value
        self.count += 1

    def compute(self):
        return self.total / self.count

    def is_best(self):
        if self.best is None:
            self.best = self.compute()
            return True
        elif self.best_type == 'min':
            if self.compute() < self.best:
                self.best = self.compute()
                return True
            else:
                return False
        elif self.best_type == 'max':
            if self.compute() > self.best:
                self.best = self.compute()
                return True
            else:
                return False
        else:
            raise ValueError('best_type must be either min or max')

def convert_to_string(tensor, index_to_word):
    """
    :param tensor: (B, L)
    :param index_to_word: dict
    :return:
    """
    tensor = tensor.tolist()
    string = ''
    for i in tensor:
        if i in [0, 2]:
            continue
        if i == 3:
            break
        string += index_to_word[i] + ' '

    return string.strip()

def compute_loss(logit, gt, seq_len):

    p = torch.nn.functional.log_softmax(logit, dim=-1)
    p = torch.gather(p, dim=-1, index=gt.unsqueeze(-1)).squeeze(-1)

    l_2d = l.repeat_interleave(torch.max(seq_len)).reshape(seq_len.shape[0], -1)
    mask = torch.arange(torch.max(seq_len)) < seq_len.view(-1, 1)
    mask_matrix = l_2d * mask / seq_len.view(-1, 1)

    # Compute Loss as cross entropy
    return -torch.sum(p * mask_matrix) / torch.sum(l)

# Setup Training Loop
train_loss, val_loss = AverageMeter(), AverageMeter()
val_wer = AverageMeter(best=True, best_type='max')
for i in range(train_params['epochs']):
    print("Epoch: ", i)
    for x, y, l in tqdm(dataloader_train):
        # Get Maximum length of a sequence in the batch, and use it to trim the output of the model
        # y.shape is (B, MAX_LEN) and x.shape is (B, L ,V) which is to be trimmed
        max_len = y.shape[1]
        logit = model(x, target=y)[:, :max_len, :] # (B, L, V)

        # Compute Loss
        loss = compute_loss(logit, y, l)

        # Backpropagation with clipped gradients
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_params['clip_grad_norm'])
        optimizer.step()

        optimizer.zero_grad()


        # Update loss
        train_loss.update(loss.item())

    # scheduler.step()
    print(f'\tTraining Loss during epoch {i}: {train_loss.compute()}')

    with torch.no_grad():
        model.eval()
        for x, y, l in dataloader_val:
            # Set model to eval mode
            max_len = y.shape[1]
            logit = model(x)[:, :max_len, :]

            # Compute Loss as cross entropy
            loss = compute_loss(logit, y, l)
            # loss = criterion(logit.contiguous().view(-1, logit.size(-1)), y.contiguous().view(-1))
            # loss = torch.nn.functional.cross_entropy(p[:,:max_len,:], torch.nn.functional.one_hot(y, model.config['vocab_size']).float())
            # val_loss.update(loss.item())

            # Computer WER
            y_pred = torch.argmax(logit, dim=1).detach().cpu().numpy()
            y_pred = [convert_to_string(y_pred[i, :], dataset.index_to_word) for i in range(y_pred.shape[0])]
            y_true = y.detach().cpu().numpy()
            y_true = [convert_to_string(y_true[i, :], dataset.index_to_word) for i in range(y_true.shape[0])]

            # Update Val_WER
            wer.update(y_pred, y_true)
            val_wer.update(wer.compute())
            wer.reset()

    print(f'\tValidation WER during epoch {i}: {val_wer.compute()}')
    print(f'\tValidation Loss during epoch {i}: {val_loss.compute()}')

    # Save model if val_wer is best
    if val_wer.is_best():
        model.save(best=True)

    # Reset AverageMeters
    train_loss.reset()
    val_loss.reset()
    val_wer.reset()

    # Save model
    model.save(iteration=i)
from utils.datasets import ImageDataset, collate_fn, convert_to_string
from utils.global_params import BASE_CONFIG, TR_IMAGE_SIZE, BATCH_SIZE, CROHME_TRAIN, VOCAB_LOC
from torch.utils.data import DataLoader, random_split
from torcheval.metrics import WordErrorRate
import torch
from models import VanillaWAP
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

torch.manual_seed(0)

SOS_INDEX = 0
EOS_INDEX = 1

# Define Dataset
# train_data_csv = pd.read_csv(CROHME_TRAIN + '/train.csv')  # Location of the generated dataset
train_data_csv = pd.read_csv(CROHME_TRAIN + '/wap_dataset.csv', sep='\t')  # Location

# train_data_csv = pd.read_csv(CROHME_TRAIN + '/train_caption.txt', sep="\t", names=['image_loc', 'label'])
# train_data_csv['image_loc'] = train_data_csv.apply(lambda row: f'{CROHME_TRAIN}/off_image_train/{row[
# "image_loc"]}_0.bmp', axis=1) train_data_csv.to_csv(CROHME_TRAIN + '/wap_dataset.csv', sep='\t')

# Define transforms
transform = transforms.Compose([transforms.ToTensor()])
dataset = ImageDataset(train_data_csv['image_loc'], train_data_csv['label'], VOCAB_LOC, device=BASE_CONFIG['DEVICE'],
                       transform=transform)

# Model
model = VanillaWAP(BASE_CONFIG)

# Training Constructs
train_params = BASE_CONFIG['train_params']
optimizer = torch.optim.AdamW(model.parameters(), lr=train_params['lr'], weight_decay=train_params['weight_decay'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=train_params['lr_decay_step'],
                                            gamma=train_params['lr_decay'])

# Evaluation Constructs
wer = WordErrorRate(device=BASE_CONFIG['DEVICE'])

# Define dataloader
generator = torch.Generator().manual_seed(train_params['random_seed'])
train, val = random_split(dataset, [0.8, 0.2], generator=generator)
dataloader_train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dataloader_val = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


# Define Expression Rate
def compute_expr_rate(pred, gt):
    correct = 0
    for i in range(len(pred)):
        pred[i] = pred[i].replace(' ', '')
        gt[i] = gt[i].replace(' ', '')
        if pred[i] == gt[i]:
            correct += 1

    return correct / len(pred)


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


def compute_loss(logit, gt, seq_len, mask):
    logp = torch.nn.functional.log_softmax(logit, dim=-1)
    div = torch.gather(logp, dim=-1, index=gt.unsqueeze(-1)).squeeze(-1)
    return -torch.mean(torch.sum(div * label_mask, dim=-1))


# Setup Training Loop
train_loss, val_loss = AverageMeter(), AverageMeter()
val_wer = AverageMeter(best=True, best_type='max')
val_expr = AverageMeter(best=True, best_type='max')
losses, word_er, expr_r = [], [], []

j = 0
for i in range(train_params['epochs']):
    print("Epoch: ", i)
    model.train()
    for x, x_mask, y, l, label_mask in tqdm(dataloader_train):
        # Get Maximum length of a sequence in the batch, and use it to trim the output of the model
        # y.shape is (B, MAX_LEN) and x.shape is (B, L ,V) which is to be trimmed
        logit = model(x, mask=x_mask, target=y)

        # Compute Loss
        loss = compute_loss(logit, y, l, label_mask)

        # Backpropagation with clipped gradients
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_params['clip_grad_norm'])
        optimizer.step()

        optimizer.zero_grad()

        # Update loss
        train_loss.update(loss.item())
        j += 1
        if j < 0:
            break

    # scheduler.step()
    print(f'\tTraining Loss during epoch {i}: {train_loss.compute()}')
    print(f'Computing Validation WER Now...')
    with torch.no_grad():
        model.eval()
        for x, x_mask, y, l, label_mask in tqdm(dataloader_val):
            # Set model to eval mode
            y_pred = model.translate(x, mask=x_mask)

            # Computer WER
            y_pred = [convert_to_string(y_pred[i, :], dataset.index_to_word) for i in range(y_pred.shape[0])]
            y_true = y.detach().cpu().numpy()
            y_true = [convert_to_string(y_true[i, :], dataset.index_to_word) for i in range(y_true.shape[0])]

            # Update Val_WER
            wer.update(y_pred, y_true)
            val_wer.update(wer.compute())
            val_expr.update(compute_expr_rate(y_pred, y_true))

    print(f'\tValidation WER during epoch {i}: {val_wer.compute()}')
    print(f'\tValidation Expression Rate during epoch {i}: {val_expr.compute()}')
    # print(f'\tValidation Loss during epoch {i}: {val_loss.compute()}')

    # Save model if val_wer is best
    if val_wer.is_best() or val_expr.is_best():
        model.save(best=True)

    losses.append(train_loss.compute())
    word_er.append(val_wer.compute())
    expr_r.append(val_expr.compute())

    # Reset AverageMeters
    train_loss.reset()
    val_wer.reset()
    wer.reset()
    val_expr.reset()

    # Save model
    model.save(iteration=i)

x = torch.range(0, len(losses))
plt.plot(x, losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('loss.png')
plt.close()

plt.plot(x, word_er, label='Word Error Rate')
plt.plot(x, expr_r, label='Expression Rate')
plt.title('Validation Performance')
plt.xlabel('Epochs')
plt.ylabel('Performance')
plt.legend()
plt.savefig('performance.png')
plt.close()

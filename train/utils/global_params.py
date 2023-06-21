import os, re, torch

# Dataset paths
ROOT_LOC = re.findall('(.+/ImageToLatex).*', os.path.abspath(__file__))[0]
CROHME_PATH = os.path.join(ROOT_LOC, 'data/CROHME')
CROHME_TRAIN = os.path.join(CROHME_PATH, 'train')
CROHME_VAL = os.path.join(CROHME_PATH, 'val')
VOCAB_LOC = CROHME_TRAIN + '/tex_symbols.csv'

# Image size Original (OG) vs Transformed (TR)
OG_IMG_SIZE = (512, 512)
TR_IMAGE_SIZE = (128, 128)

# CNN input dimension
CNN_INPUT_DIM = [128, 128]

# Compute VOCAB SIZE
with open(VOCAB_LOC, 'r') as f:
    VOCAB_SIZE = len(f.readlines())

# Training Parameters
BATCH_SIZE = 32

# Base CONFIG
BASE_CONFIG = {
    'root_loc': ROOT_LOC,
    'num_layers': 16,
    'input_dim': CNN_INPUT_DIM,
    'input_channels': 1,
    'num_features_map': [32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128],
    'feature_kernel_size': [3] * 16,
    'feature_kernel_stride': [1] * 16,
    'feature_padding': [1] * 16,
    'feature_pooling_kernel_size': [None, None, None, (2, 2), None, None, None, (2, 2),
                                    None, None, None, (2, 2), None, None, None, (2, 2)],
    'feature_pooling_stride': [None, None, None, (2, 2), None, None, None, (2, 2),
                               None, None, None, (2, 2), None, None, None, (2, 2)],
    'batch_norm':[True] * 16,
    'DEVICE': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'hidden_dim': 256,
    'cell_dim': 64,
    'vocab_size': VOCAB_SIZE + 4,
    'embedding_dim': 100,
    'LSTM_bidirectional': False,
    'LSTM_num_layers': 1,
    'dropout': 0.2,
    'max_len': 100,
    'train_params': {
        'random_seed': 42,
        'lr': 0.0002,
        'epochs': 10,
        'lr_decay': 0.5,
        'clip_grad_norm': 100,
        'lr_decay_step': 10,
        'print_every': 100,
        'save_every': 1,
        'save_loc': './checkpoints/',
        'load_loc': None,
        'load': False,
        'load_epoch': 0,
        'load_step': 0,
        'load_best': False,
        'load_iter': 20,
        'load_best_epoch': 0,
        'batch_size': BATCH_SIZE,
    }
}

BASE_CONFIG['output_dim'] = BASE_CONFIG['input_dim']
for i in range(BASE_CONFIG['num_layers']):
    dim, p, s, k = (BASE_CONFIG['output_dim'], BASE_CONFIG['feature_padding'][i],
                        BASE_CONFIG['feature_kernel_stride'][i], BASE_CONFIG['feature_kernel_size'][i])

    dim[0] = (dim[0] + 2 * p - k) // s + 1
    dim[1] = (dim[1] + 2 * p - k) // s + 1

    if BASE_CONFIG['feature_pooling_kernel_size'][i]:
        ps, pk = BASE_CONFIG['feature_pooling_stride'][i], BASE_CONFIG['feature_pooling_kernel_size'][i]
        dim[0] = (dim[0] - pk[0]) // ps[0] + 1
        dim[1] = (dim[1] - pk[1]) // ps[1] + 1

    BASE_CONFIG['output_dim'] = dim


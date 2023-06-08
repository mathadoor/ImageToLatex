import os, re, torch

# Dataset paths
CROHME_PATH = os.path.abspath(__file__)
CROHME_PATH = re.findall('(.+/ImageToLatex).*', CROHME_PATH)[0]
CROHME_PATH = os.path.join(CROHME_PATH, 'data/CROHME')
CROHME_TRAIN = os.path.join(CROHME_PATH, 'train')
CROHME_VAL = os.path.join(CROHME_PATH, 'val')

# Image size Original (OG) vs Transformed (TR)
OG_IMG_SIZE = (512, 512)
TR_IMAGE_SIZE = 32

# CNN input dimension
CNN_INPUT_DIM = [32, 32]

# Base Model Parameters
BASE_CONFIG = {
    'num_layers': 6,
    'input_dim': CNN_INPUT_DIM,
    'input_channels': 1,
    'num_features_map': [64, 128, 256, 256, 512, 512],
    'feature_kernel_size': [3, 3, 3, 3, 3, 3],
    'feature_kernel_stride': [1, 1, 1, 1, 1, 1],
    'feature_padding': [1, 1, 1, 1, 1, 1],
    'feature_pooling_kernel_size': [(2, 2), (2, 2), None, (1, 2), (2, 1), None],
    'feature_pooling_stride': [(2, 2), (2, 2), None, (1, 2), (2, 1), None],
    'batch_norm':[False, False, True, False, True, True],
    'DEVICE': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
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


# Training Parameters
BATCH_SIZE = 32
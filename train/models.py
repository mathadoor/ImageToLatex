# Define the model architectures here
import torch
from torch import nn, optim
import torch.nn.functional as F

BASE_CONFIG = {
    'num_layers': 3,
    'layer_dim_config': [32, 64, 128],
    'kernel_size': 3,
    'num_classes': 10,
    'DEVICE': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}
class baseWatcher:
    def __init__(self, config):
        if not config:
            self.config = BASE_CONFIG


    def train(self):
        pass

    def test(self):
        pass

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

class baseParser:
    def __init__(self, config):
        self.config = config

    def parse(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

class VanilleCNNWatcher(baseWatcher):
    def __init__(self, config):
        super().__init__(config)
        self.generate_model()

    def generate_model(self):
        '''
        Generate the model based on the config
        :return: None
        '''
        self.model = nn.Sequential()
        layer_dims = self.config['layer_dim_config'] + [self.config['num_classes']]
        kernel_size = self.config['kernel_size']
        for i in range(self.config['num_layers'] + 1):
            self.model.add_module('conv{}'.format(i), nn.Conv2d(layer_dims[i], layer_dims[i+1], kernel_size))
            self.model.add_module('relu{}'.format(i), nn.ReLU())
            self.model.add_module('maxpool{}'.format(i), nn.MaxPool2d(2, 2))
        self.model.to(self.config['DEVICE'])

    def train(self):
        pass

    def test(self):
        pass

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

class VanilleCNNParser(baseParser):
    def __init__(self, config):
        super().__init__(config)

    def parse(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
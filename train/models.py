# Define the model architectures here
from torch import nn
from train.utils.global_params import *

class VanilleWAP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.watcher = None
        self.tokenizer = None
        self.positional_encoder = None
        self.parser = None

        if not config:
            self.config = BASE_CONFIG

        self.generate_watcher()
        self.generate_embedder()
        self.generate_parser()

    def generate_watcher(self):
        '''
        Generate the model based on the config
        :return: None
        '''
        self.watcher = nn.Sequential()
        layer_dims = [CNN_INPUT_DIM] + self.config['num_features_map']
        for i in range(self.config['num_layers'] + 1):
            self.watcher.add_module('conv{}'.format(i),
                                    nn.Conv2d(layer_dims[i], layer_dims[i+1],
                                            self.config['feature_kernel_size'][i],
                                            self.config['feature_kernel_stride'][i],
                                            self.config['feature_padding'][i]))

            if self.config['batch_norm'][i]:
                self.watcher.add_module('batchnorm{}'.format(i), nn.BatchNorm2d(layer_dims[i + 1]))

            self.watcher.add_module('relu{}'.format(i), nn.ReLU())

            if self.config['feature_pooling_kernel_size'][i]:
                self.watcher.add_module('maxpool{}'.format(i),
                                        nn.MaxPool2d(self.config['feature_pooling_kernel_size'][i],
                                                   self.config['feature_pooling_stride'][i]))

        self.watcher.to(self.config['DEVICE'])

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
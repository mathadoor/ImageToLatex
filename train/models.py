# Define the model architectures here
import torch
from torch import nn

class VanillaWAP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.watcher = None
        self.tokenizer = None
        self.positional_encoder = None
        self.parser = None
        self.config = config

        self.generate_watcher()
        self.generate_tokenizer()
        self.generate_positional_encoder()
        self.generate_parser()

    def forward(self, x):
        # CNN Feature Extraction
        x = self.watcher(x)

        # Positional Encoding
        x = x + self.positional_encoder

        # RNN Decoder
        # x = self.parser(x)

        return x
    def generate_watcher(self):
        '''
        Generate the model based on the config
        :return: None
        '''
        self.watcher = nn.Sequential()
        layer_dims = [self.config['input_channels']] + self.config['num_features_map']
        for i in range(self.config['num_layers']):
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

    def generate_positional_encoder(self):
        '''
        Adds positional encodings to the
        :return:
        '''
        x, y = torch.arange(self.config['output_dim'][0]), torch.arange(self.config['output_dim'][1])
        i, j = torch.arange(self.config['num_features_map'][-1] // 4), torch.arange(self.config['num_features_map'][-1] // 4)
        D = self.config['num_features_map'][-1]

        pe = torch.zeros((D, self.config['output_dim'][0], self.config['output_dim'][1]))

        x_i = torch.einsum("i, j, k -> ijk", 10000 ** (4 * i / D), x, torch.ones_like(y))
        y_i = torch.einsum("i, j, k -> ijk", 10000 ** (4 * j / D), torch.ones_like(x), y)

        x, y = x.unsqueeze(1), y.unsqueeze(0)
        i, j = i.unsqueeze(-1).unsqueeze(-1), j.unsqueeze(-1).unsqueeze(-1)

        pe[2 * i, x, y]              = torch.sin(x_i)
        pe[2 * i + 1, x, y]          = torch.cos(x_i)
        pe[2 * j + D // 2, x, y]     = torch.sin(y_i)
        pe[2 * j + 1 + D // 2, x, y] = torch.cos(y_i)

        self.positional_encoder = pe

    def generate_tokenizer(self):
        '''
        Generates the tokenizer
        :return:
        '''
        pass

    def generate_parser(self):
        '''
        Generates the parser
        :return:
        '''
        pass

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

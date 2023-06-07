# Define the model architectures here
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
        # x = x + self.positional_encoder(x)

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
        pass

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

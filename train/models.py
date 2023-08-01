# Define the model architectures here
import torch
from torch import nn
import os
class VanillaWAP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.watcher = None
        self.embedder = None
        self.positional_encoder = None
        self.parser = None
        self.config = config


        self.generate_watcher()
        self.generate_positional_encoder()
        self.generate_embedder()
        self.generate_parser()

        # Generate the model based on the config. If load is True, load the model from the path
        if config['train_params']['load']:
            self.load()

    def forward(self, x, target=None):
        # CNN Feature Extraction
        x = self.watcher(x)

        # Positional Encoding
        x = x + self.positional_encoder
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        x = x.permute(0, 2, 1)

        # RNN Decoder

        y = 2 * torch.ones((x.shape[0], 1)).long()
        logit = torch.zeros((x.shape[0], self.config['max_len'], self.config['vocab_size'] ))

        logit[:, 0, :] = 2
        # While all y are not index = 3 and max length is not reached
        o_t, c_t, h_t = None, None, None
        for i in range(1, self.config['max_len']):

            if target is not None:
                if i >= target.shape[1]:
                    break
                y = target[:, i].unsqueeze(1)

            # Embedding
            logit_t, h_t = self.parse(x, y, h_t)
            logit[:, i, :] = logit_t.squeeze()
            y = torch.argmax(logit_t.squeeze(), dim=1)

        return logit

    def generate_watcher(self):
        """
        Generate the model based on the config
        :return: None
        """
        # Sequential model
        self.watcher = nn.Sequential()

        # Kernel Dimension of each layer
        layer_dims = [self.config['input_channels']] + self.config['num_features_map']
        for i in range(self.config['num_layers']):
            # Convolutional Layer
            self.watcher.add_module('conv{}'.format(i),
                                    nn.Conv2d(layer_dims[i], layer_dims[i+1],
                                            self.config['feature_kernel_size'][i],
                                            self.config['feature_kernel_stride'][i],
                                            self.config['feature_padding'][i]))

            # Batch Normalization if required
            if self.config['batch_norm'][i]:
                self.watcher.add_module('batchnorm{}'.format(i), nn.BatchNorm2d(layer_dims[i + 1]))

            # Activation Function
            self.watcher.add_module('relu{}'.format(i), nn.ReLU())

            # Max Pooling if required
            if self.config['feature_pooling_kernel_size'][i]:
                self.watcher.add_module('maxpool{}'.format(i),
                                        nn.MaxPool2d(self.config['feature_pooling_kernel_size'][i],
                                                   self.config['feature_pooling_stride'][i]))

        self.watcher.to(self.config['DEVICE'])

    def generate_positional_encoder(self):
        """
        Generate 2-D Positional Encoding as per https://arxiv.org/pdf/1908.11415.pdf
        :return:
        """
        x, y = torch.arange(self.config['output_dim'][0]), torch.arange(self.config['output_dim'][1], requires_grad=False)
        i, j = torch.arange(self.config['num_features_map'][-1] // 4), torch.arange(self.config['num_features_map'][-1] // 4, requires_grad=False )
        D = self.config['num_features_map'][-1]

        pe = torch.zeros((D, self.config['output_dim'][0], self.config['output_dim'][1]), requires_grad=False)

        x_i = torch.einsum("i, j, k -> ijk", 10000 ** (4 * i / D), x, torch.ones_like(y))
        y_i = torch.einsum("i, j, k -> ijk", 10000 ** (4 * j / D), torch.ones_like(x), y)

        x, y = x.unsqueeze(1), y.unsqueeze(0)
        i, j = i.unsqueeze(-1).unsqueeze(-1), j.unsqueeze(-1).unsqueeze(-1)

        pe[2 * i, x, y]              = torch.sin(x_i)
        pe[2 * i + 1, x, y]          = torch.cos(x_i)
        pe[2 * j + D // 2, x, y]     = torch.sin(y_i)
        pe[2 * j + 1 + D // 2, x, y] = torch.cos(y_i)

        self.positional_encoder = pe

    def generate_embedder(self):
        """
        Generates the embedder
        :return:
        """
        self.embedder = nn.Embedding(self.config['vocab_size'], self.config['embedding_dim'], padding_idx=0)

    def generate_parser(self):
        """
        Generates the parser
        :return:
        """

        D = self.config['num_features_map'][-1]
        L = self.config['output_dim'][0] * self.config['output_dim'][1]

        self.parser = nn.ModuleDict({
            # Attention Weights
            "U_a" : nn.Linear(D, L),
            "W_a" : nn.Linear(self.config['hidden_dim'], L),
            "nu_a" : nn.Linear(L, 1, bias=False),

            # Hidden Layer Weights
            "U_hz" : nn.Linear(self.config['hidden_dim'], self.config['hidden_dim']),
            "U_hr" : nn.Linear(self.config['hidden_dim'], self.config['hidden_dim']),
            "U_rh" : nn.Linear(self.config['hidden_dim'], self.config['hidden_dim']),

            # Input Layer Weights
            "W_yz" : nn.Linear(self.config['embedding_dim'], self.config['hidden_dim']),
            "W_yr" : nn.Linear(self.config['embedding_dim'], self.config['hidden_dim']),
            "W_yh" : nn.Linear(self.config['embedding_dim'], self.config['hidden_dim']),

            # Context Layer Weights
            "C_cz" : nn.Linear(D, self.config['hidden_dim']),
            "C_cr" : nn.Linear(D, self.config['hidden_dim']),

            # Output Layer Weights
            "W_c" : nn.Linear(D, self.config['embedding_dim']),
            "W_h" : nn.Linear(self.config['hidden_dim'], self.config['embedding_dim']),
            "W_o" : nn.Linear(self.config['embedding_dim'], self.config['vocab_size']),

        })

        self.parser.to(self.config['DEVICE'])

    def parse(self, x, y, h_t_1=None):
        """
        x is of shape (batch_size, num_features_map[-1], 1, output_dim[0]*output_dim[1]) - (B, D, 1, L)
        y is of shape (batch_size, vocab) - (B, V)
        h_t_1 is of shape (batch_size, hidden_dim) - (B, H)
        o_t_1 is of shape (batch_size, hidden_dim) - (B, H)
        """

        # Compute the initial hidden and cell states of the forward and reverse LSTMs
        if h_t_1 is None:
            h_t_1 = torch.zeros(x.shape[0], self.config['hidden_dim'], device=self.config['DEVICE'])

        ey_t_1 = self.embedder(y).squeeze() # (B, E)

        # Compute the attention weights and context vector
        e_t = torch.tanh(self.parser['U_a'](x) + self.parser['W_a'](h_t_1).unsqueeze(1)) # (B, L, L)
        e_t = self.parser['nu_a'](e_t).squeeze() # (B, L)
        alpha_t  = torch.softmax(e_t, dim=-1) # (B, L)
        ct = torch.einsum("bl, bld -> bd", alpha_t, x) # (B, D)

        # Forward Pass through GRU
        zt = torch.sigmoid(self.parser['W_yz'](ey_t_1) + self.parser['U_hz'](h_t_1) + self.parser['C_cz'](ct)) # (B, H)
        rt = torch.sigmoid(self.parser['W_yr'](ey_t_1) + self.parser['U_hr'](h_t_1) + self.parser['C_cr'](ct)) # (B, H)

        rt_ht = rt * h_t_1 # (B, H)
        ht_tilde = torch.tanh(self.parser['W_yh'](ey_t_1) + self.parser['U_rh'](rt_ht) + self.parser['C_cz'](ct)) # (B, H)

        h_t = (1 - zt) * h_t_1 + zt * ht_tilde # (B, H)

        # Compute the output
        combined = self.parser['W_c'](ct) + self.parser['W_h'](h_t) # (B, E)
        o_t = self.parser['W_o'](combined + ey_t_1) # (B, V)

        return o_t, h_t

    def predict(self, x):
        pass

    def save(self, iteration=None, best=False):
        """
        Saves the model every self.config.train_params['save_every'] iterations in the directory specified by
        'save_loc' key in the config file
        :param best:
        :param iteration:
        :return:
        """
        if best:
            save_loc = os.path.join(self.config['root_loc'], self.config['train_params']['save_loc'])
            torch.save(self.state_dict(), os.path.join(save_loc, 'model_best.pth'))
            return

        if iteration % self.config['train_params']['save_every'] == 0:
            save_loc = os.path.join(self.config['root_loc'], self.config['train_params']['save_loc'])
            torch.save(self.state_dict(), os.path.join(save_loc, 'model_{}.pth'.format(iteration)))
            return

    def load(self):
        # load the model
        path = os.path.join(self.config['root_loc'], self.config['train_params']['save_loc'])
        if self.config['train_params']['load_best']:
            path = os.path.join(path, 'model_best.pth')
        else:
            path = os.path.join(path, 'model_{}.pth'.format(self.config['train_params']['load_iter']))
        self.load_state_dict(torch.load(path))

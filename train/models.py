# Define the model architectures here
import torch
from torch import nn
import os

SOS_INDEX = 0
EOS_INDEX = 1

class VanillaWAP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.watcher = None
        self.embedder = None
        self.positional_encoder = None
        self.parser = None
        self.config = config

        self.generate_watcher()
        # self.generate_positional_encoder()
        self.generate_embedder()
        self.generate_parser()

        # Generate the model based on the config. If load is True, load the model from the path
        if config['train_params']['load']:
            self.load()

    def forward(self, x, mask, target, gen_viz=False):
        # CNN Feature Extraction
        max_len = target.shape[1]
        x, feature_mask = self.watch(x, mask)

        # # Positional Encoding
        # # x = x + self.positional_encoder
        # x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        # x = x.permute(0, 2, 1)
        # feature_mask = torch.reshape(feature_mask, (feature_mask.shape[0], feature_mask.shape[1], -1))
        # feature_mask = feature_mask.permute(0, 2, 1)

        # RNN Decoder

        y = SOS_INDEX * torch.ones((x.shape[0], 1)).long().to(self.config['DEVICE'])
        logit = torch.zeros((x.shape[0], max_len, self.config['vocab_size'])).to(self.config['DEVICE'])
        alpha_past = torch.zeros_like(feature_mask).to(self.config['DEVICE'])

        # logit[:, 0, 2] = 0
        # While all y are not index = EOS_INDEX and max length is not reached
        h_t = None
        for i in range(max_len):

            if i > 0:
                if i >= target.shape[1]:
                    break
                y = target[:, i - 1].unsqueeze(1)

            # Embedding
            logit_t, h_t, alpha_past, alpha = self.parse(x, y, h_t, feature_mask, alpha_past)
            logit[:, i, :] = logit_t.squeeze()
            # y = torch.argmax(logit_t.squeeze(), dim=1)

            # if all y are index = EOS_INDEX, break
            # if torch.all(y == EOS_INDEX):
            #     break

        return logit

    def translate(self, x, beam_width=10, mask=None):
        """
        Translate the input image to the corresponding latex
        :return:
        """
        # CNN Feature Extraction
        max_len = self.config['max_len']
        x, feature_mask = self.watch(x, mask)

        # RNN Decoder

        y = SOS_INDEX * torch.ones((x.shape[0], 1)).long().to(self.config['DEVICE'])
        ret = []
        ret_alphas = []
        alpha_past = torch.zeros_like(feature_mask).to(self.config['DEVICE'])

        # logit[:, 0, 2] = 0
        # While all y are not index = EOS_INDEX and max length is not reached
        h_t = None
        for i in range(max_len):

            # Embedding
            logit_t, h_t, alpha_past, alpha = self.parse(x, y, h_t, feature_mask, alpha_past)
            y = torch.argmax(logit_t.squeeze(), dim=-1)
            ret.append(y)
            ret_alphas.append(alpha)

            # if all y are index = EOS_INDEX, break
            if torch.all(y == EOS_INDEX):
                break
        return torch.stack(ret, dim=-1), ret_alphas
    def generate_watcher(self):
        """
        Generate the model based on the config
        :return: None
        """
        # Sequential model
        self.watcher = nn.Sequential()

        # Kernel Dimension of each layer
        # layer_dims = [self.config['input_channels']] + self.config['num_features_map']
        for i in range(self.config['num_blocks']):
            curr_block = nn.Sequential()
            for j in range(self.config['num_layers'][i]):
                # Convolutional Layer
                curr_layer = nn.Sequential()
                if j == 0:
                    if i == 0:
                        cin = self.config['input_channels']
                    else:
                        cin = self.config['num_features_map'][i - 1][-1]
                else:
                    cin = self.config['num_features_map'][i][-1]
                cout = self.config['num_features_map'][i][j]

                curr_layer.add_module(f'conv2d', nn.Conv2d(cin, cout,
                                                           self.config['feature_kernel_size'][i][j],
                                                           self.config['feature_kernel_stride'][i][j],
                                                           self.config['feature_padding'][i][j]))

                # Dropout Layer
                if self.config['conv_dropout'][i][j] != 0:
                    curr_layer.add_module('conv_dropout2d', nn.Dropout2d(p=self.config['conv_dropout'][i][j]))

                # Activation Function
                curr_layer.add_module('relu{}', nn.ReLU())

                # Max Pooling if required
                if self.config['feature_pooling_kernel_size'][i][j]:
                    curr_layer.add_module('maxpool2d', nn.MaxPool2d(self.config['feature_pooling_kernel_size'][i][j],
                                                                    self.config['feature_pooling_stride'][i][j]))
                # Batch Normalization if required
                if self.config['batch_norm'][i][j]:
                    curr_layer.add_module('batchnorm2d', nn.BatchNorm2d(cout))

                curr_block.add_module(f'layer{j + 1}', curr_layer)
            self.watcher.add_module(f'block{i + 1}', curr_block)

        self.watcher.to(self.config['DEVICE'])

    def generate_embedder(self):
        """
        Generates the embedder
        :return:
        """
        self.embedder = nn.Embedding(self.config['vocab_size'], self.config['embedding_dim'], padding_idx=0).to(
            self.config['DEVICE'])

    def generate_parser(self):
        """
        Generates the parser
        :return:
        """

        D = self.config['num_features_map'][-1][-1]

        self.parser = nn.ModuleDict({

            # Initial Conversion
            "W_2h": nn.Linear(D, self.config['hidden_dim']),

            # New Weights
            # Computing Preactivation1
            "W" : nn.Linear(self.config['embedding_dim'], 2 * self.config['hidden_dim']),
            "U" : nn.Linear(self.config['hidden_dim'], 2 * self.config['hidden_dim'], bias=False),

            # Computing Preactivation-x1
            "Wx" : nn.Linear(self.config['embedding_dim'], self.config['hidden_dim']),
            "Ux" : nn.Linear(self.config['hidden_dim'], self.config['hidden_dim'], bias=False),

            # Computing Non-Linear Preactivation1 and Preactivation-x1
            "U_nl" : nn.Linear(self.config['hidden_dim'], 2 * self.config['hidden_dim']),
            "Ux_nl" : nn.Linear(self.config['hidden_dim'], self.config['hidden_dim']),

            # Computing Context to GRU
            "Wc" : nn.Linear(D, 2 * self.config['hidden_dim'], bias=False),
            "Wcx" : nn.Linear(D, self.config['hidden_dim'], bias=False),

            # Combined Attention
            "W_comb_att" : nn.Linear(self.config['hidden_dim'], self.config['attention_dim'], bias=False),

            # Context Translation to Attention Dimension
            "Wc_att" : nn.Conv2d(D, self.config['attention_dim'], 1, padding='valid'),

            # Attention Weights
            "U_att" : nn.Linear(self.config['attention_dim'], 1, bias=False),

            # Coverage Weights
            "conv_q" : nn.Conv2d(1, self.config['coverage_dim'], (5, 5), padding='same'),
            "conv_uf" : nn.Linear(self.config['coverage_dim'], self.config['attention_dim']),

            # Output Layer Weights
            "W_c": nn.Linear(D, self.config['embedding_dim']),
            "W_h": nn.Linear(self.config['hidden_dim'], self.config['embedding_dim']),
            "W_yo": nn.Linear(self.config['embedding_dim'], self.config['embedding_dim']),
            "W_o": nn.Linear(self.config['embedding_dim'] // 2, self.config['vocab_size']),

        })

        self.parser.to(self.config['DEVICE'])

    def parse(self, x, y, h_t_1=None, feature_mask=None, alpha_past=None, alpha=None):
        """
        x is of shape (batch_size, num_features_map[-1], 1, output_dim[0]*output_dim[1]) - (B, D, 1, L)
        y is of shape (batch_size, vocab) - (B, V)
        h_t_1 is of shape (batch_size, hidden_dim) - (B, H)
        o_t_1 is of shape (batch_size, hidden_dim) - (B, H)
        """

        # Compute the initial hidden
        if h_t_1 is None:
            ctx_mean = torch.einsum('...hw, ...hw -> ...', feature_mask, x) / torch.einsum('...hw -> ...', feature_mask)
            h_t_1 = torch.tanh(self.parser['W_2h'](ctx_mean))  # (B, H)

        # Compute the attention weights and context vector
        state_below = self.embedder(y).squeeze()  # (B, E)

        # Translate Embedding to hidden dimension
        state_below_ = self.parser['W'](state_below)  # (B, 2H)
        state_belowx = self.parser['Wx'](state_below)  # (B, H)

        # Compute the preactivation and split it into r and u of size (B, H)
        preact = torch.sigmoid(self.parser['U'](h_t_1) + state_below_)   # (B, 2H)
        r, u = preact[..., :self.config['hidden_dim']], preact[..., self.config['hidden_dim']:]   # (B, H)

        # Compute preactivation-x1
        preactx = r * self.parser['Ux'](h_t_1) + state_belowx  # (B, H)
        h_tilde = torch.tanh(preactx)  # (B, H)
        h1 = u * h_t_1 + (1. - u) * h_tilde  # (B, H)

        # Compute the Needed Context

        # Project the Context and the state to the attention dimension
        pctx_ = self.parser['Wc_att'](x)  # (B, A, Height, Width)
        pstate_ = self.parser['W_comb_att'](h1).unsqueeze(-1).unsqueeze(-1)  # (B, A, 1, 1)

        # Compute the coverage
        if len(alpha_past.shape) == 5:
            cover_F = []
            for _ in range(alpha_past.shape[0]):
                cover_F.append(self.parser['conv_q'](alpha_past[_]))  # (B, A, Height, Width)
            cover_F = torch.stack(cover_F, dim=0)
            # alpha_shape = alpha_past.shape
            # cover_F = self.parser['conv_q'](alpha_past.reshape(-1, alpha_shape[2], alpha_shape[3], alpha_shape[4]))  # (B, A, Height, Width)
            # cover_F = cover_F.reshape(alpha_shape[0], -1, alpha_shape[2], alpha_shape[3], alpha_shape[4])
            cover_vector = self.parser['conv_uf'](cover_F.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)  # (B, 1, A, Height, Width)
        else:
            cover_F = self.parser['conv_q'](alpha_past)  # (B, A, Height, Width)
            cover_vector = self.parser['conv_uf'](cover_F.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B, 1, A, Height, Width)

        if len(h1.shape) == 3:
            pctx_ = pctx_.unsqueeze(0)
            if len(alpha_past.shape) != 5:
                cover_vector = cover_vector.unsqueeze(0)
                alpha_past = alpha_past.unsqueeze(0)

        # compute attention
        pctx__ = torch.tanh(pctx_ + pstate_ + cover_vector)
        pctx__ = torch.transpose(torch.transpose(pctx__, -2, -1), -3, -1)  # (B, Height, Width, A)
        alpha = torch.exp(self.parser['U_att'](pctx__).squeeze() + torch.where(feature_mask == 0, -torch.inf, 0).squeeze()) # (B, Height, Width)
        alpha_t = alpha / torch.einsum('...hw -> ...', alpha).unsqueeze(-1).unsqueeze(-1)  # (B, Height, Width)
        alpha_past = alpha_past + alpha_t.unsqueeze(-3)  # (B, 1, Height, Width)

        # Compute Context
        ct = torch.einsum('...hw, ...dhw -> ...d', alpha_t, x)  # (B, D) or (Beam_width, B, D)

        # Layer two Computation
        preactivation2 = torch.sigmoid(self.parser['U_nl'](h1) + self.parser['Wc'](ct))   # (B, 2*H)
        r2, u2 = preactivation2[..., :self.config['hidden_dim']], preactivation2[..., self.config['hidden_dim']:]   # (B, H)

        h_tilde = torch.tanh(r2 * self.parser['Ux_nl'](h1) + self.parser['Wcx'](ct))  # (B, H)

        ht = u2 * h1 + (1. - u2) * h_tilde  # (B, H)

        # Compute the output
        logit_ctx = self.parser['W_c'](ct)  # (B, E)
        logit_ht = self.parser['W_h'](ht)  # (B, E)
        logit_ey = self.parser['W_yo'](state_below)  # (B, E)

        logit = logit_ctx + logit_ht + logit_ey  # (B, E)

        # reshaped logit and max out layer
        shape = list(logit.shape)
        shape = tuple(shape[:-1] + [shape[-1] // 2, 2])
        logit = torch.max(logit.reshape(shape), dim=-1)[0]

        o_t = self.parser['W_o'](logit)  # (B, V)

        return o_t, ht, alpha_past, alpha

    def visualize(self, images, mask, labels):
        """
        Visualize the attention maps
        :param images:
        :param mask:
        :param labels:
        :return:
        """
        num_images = images.shape[0]
        for i in range(num_images):
            image = images[i]
            label = labels[i]
            mask = mask[i]

            # Shrink the image where mask is 0
            image = image * mask

            # Watch
            x, mask = self.watch(image, mask)

            upsampler = nn.Upsample(size=image.shape, mode='bilinear', align_corners=True)

    def watch(self, x, mask=None):
        for i in range(self.config['num_blocks']):
            x = self.watcher[i](x)

            if mask is not None:
                mask = mask[:, :, ::2, ::2]
            if mask.shape[2] != x.shape[2] or mask.shape[3] != x.shape[3]:
                mask = mask[:, :, :x.shape[2], :x.shape[3]]
        return x, mask

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

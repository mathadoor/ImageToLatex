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

        # Positional Encoding
        # x = x + self.positional_encoder
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        x = x.permute(0, 2, 1)
        feature_mask = torch.reshape(feature_mask, (feature_mask.shape[0], feature_mask.shape[1], -1))
        feature_mask = feature_mask.permute(0, 2, 1)

        # RNN Decoder

        y = SOS_INDEX * torch.ones((x.shape[0], 1)).long().to(self.config['DEVICE'])
        logit = torch.zeros((x.shape[0], max_len, self.config['vocab_size'])).to(self.config['DEVICE'])

        # logit[:, 0, 2] = 0
        # While all y are not index = EOS_INDEX and max length is not reached
        h_t = None
        for i in range(max_len):

            if i > 0:
                if i >= target.shape[1]:
                    break
                y = target[:, i - 1].unsqueeze(1)

            # Embedding
            logit_t, h_t = self.parse(x, y, h_t, feature_mask)
            logit[:, i, :] = logit_t.squeeze()
            y = torch.argmax(logit_t.squeeze(), dim=1)

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
        x, feature_mask = self.watch(x, mask)
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        x = x.permute(0, 2, 1)
        feature_mask = torch.reshape(feature_mask, (feature_mask.shape[0], feature_mask.shape[1], -1))
        feature_mask = feature_mask.permute(0, 2, 1)

        # RNN Decoder with beam search. Keep track of the top beam_width candidates
        y = SOS_INDEX * torch.ones((x.shape[0], 1)).long().to(self.config['DEVICE'])
        ret = []

        # p, l, stop to compute the probability, length, and stop flag of each hypothesis
        hypo_p = torch.zeros((beam_width, x.shape[0], 1)).to(self.config['DEVICE'])
        hypo_l = torch.zeros((beam_width, x.shape[0], 1)).to(self.config['DEVICE'])
        hypo_stop = torch.zeros((beam_width, x.shape[0], 1)).to(self.config['DEVICE']).to(torch.bool)

        h_t = None

        for i in range(self.config['max_len']):
            # If all y are index = EOS_INDEX, break
            if torch.all(y == EOS_INDEX):
                break
            # Generate the logit for the current time step
            logit_ht, h_t = self.parse(x, y, h_t, feature_mask)

            # Get the probabilities of the top beam_width candidates

            probs = torch.log_softmax(logit_ht.squeeze(), dim=-1)
            hypo_l += ~hypo_stop

            # Get the top beam_width candidates
            if i > 0:
                probs = (probs + ~hypo_stop * hypo_p).permute(1, 0, 2)

            hypo_p, top_idx = torch.topk(probs.reshape(x.shape[0], -1), beam_width)

            # Project the top_idx to the corresponding beam_index and token_index
            beam_idx = (top_idx // self.config['vocab_size']).permute(1, 0)
            y = (top_idx % self.config['vocab_size']).permute(1, 0).unsqueeze(-1)

            # Get the hypothesis length, probability, and stop flag based on the beam_idx
            hypo_l = torch.gather(hypo_l, 0, beam_idx.unsqueeze(-1))
            hypo_stop = torch.gather(hypo_stop, 0, beam_idx.unsqueeze(-1))
            hypo_p = hypo_p.permute(1, 0).unsqueeze(-1)

            # Select the decoded sequences in ret based on the beam_idx
            if len(ret) != 0:
                b = beam_idx.unsqueeze(-1).expand(-1, -1, ret.shape[-1])
                ret = torch.gather(ret, 0, b)
                ret = torch.cat((ret, y), dim=-1)
            else:
                ret = y

            # Update the stop flag
            hypo_stop = ((y == EOS_INDEX) | hypo_stop)

            if torch.all(hypo_stop):
                break

        # Select the best candidate based on hypothesis probability / length
        best_idx = torch.argmax(hypo_p / hypo_l, dim=0).squeeze()
        ans = []
        for i, index in enumerate(best_idx):
            ans.append(ret[index, i, :].squeeze(0))

        return torch.stack(ans, dim=0)
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

    def generate_positional_encoder(self):
        """
        Generate 2-D Positional Encoding as per https://arxiv.org/pdf/1908.11415.pdf
        :return:
        """
        x, y = torch.arange(self.config['output_dim'][0]), torch.arange(self.config['output_dim'][1],
                                                                        requires_grad=False)
        i, j = torch.arange(self.config['num_features_map'][-1] // 4), torch.arange(
            self.config['num_features_map'][-1] // 4, requires_grad=False)
        D = self.config['num_features_map'][-1]

        pe = torch.zeros((D, self.config['output_dim'][0], self.config['output_dim'][1]), requires_grad=False)

        x_i = torch.einsum("i, j, k -> ijk", 10000 ** (4 * i / D), x, torch.ones_like(y))
        y_i = torch.einsum("i, j, k -> ijk", 10000 ** (4 * j / D), torch.ones_like(x), y)

        x, y = x.unsqueeze(1), y.unsqueeze(0)
        i, j = i.unsqueeze(-1).unsqueeze(-1), j.unsqueeze(-1).unsqueeze(-1)

        pe[2 * i, x, y] = torch.sin(x_i)
        pe[2 * i + 1, x, y] = torch.cos(x_i)
        pe[2 * j + D // 2, x, y] = torch.sin(y_i)
        pe[2 * j + 1 + D // 2, x, y] = torch.cos(y_i)

        self.positional_encoder = pe.to(self.config['DEVICE'])

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

            # Attention Weights
            "U_a": nn.Linear(D, self.config['attention_dim'], bias=False),
            "W_a": nn.Linear(self.config['hidden_dim'], self.config['attention_dim'], bias=False),
            "nu_a": nn.Linear(self.config['attention_dim'], 1, bias=False),

            # Hidden Layer Weights
            "U_hz": nn.Linear(self.config['hidden_dim'], self.config['hidden_dim']),
            "U_hr": nn.Linear(self.config['hidden_dim'], self.config['hidden_dim']),
            "U_rh": nn.Linear(self.config['hidden_dim'], self.config['hidden_dim']),

            # Input Layer Weights
            "W_yz": nn.Linear(self.config['embedding_dim'], self.config['hidden_dim']),
            "W_yr": nn.Linear(self.config['embedding_dim'], self.config['hidden_dim']),
            "W_yh": nn.Linear(self.config['embedding_dim'], self.config['hidden_dim']),

            # Context Layer Weights
            "C_cz": nn.Linear(D, self.config['hidden_dim']),
            "C_cr": nn.Linear(D, self.config['hidden_dim']),

            # Output Layer Weights
            "W_c": nn.Linear(D, self.config['embedding_dim']),
            "W_h": nn.Linear(self.config['hidden_dim'], self.config['embedding_dim']),
            "W_yo": nn.Linear(self.config['embedding_dim'], self.config['embedding_dim']),
            "W_o": nn.Linear(self.config['embedding_dim'] // 2, self.config['vocab_size']),

        })

        self.parser.to(self.config['DEVICE'])

    def parse(self, x, y, h_t_1=None, feature_mask=None):
        """
        x is of shape (batch_size, num_features_map[-1], 1, output_dim[0]*output_dim[1]) - (B, D, 1, L)
        y is of shape (batch_size, vocab) - (B, V)
        h_t_1 is of shape (batch_size, hidden_dim) - (B, H)
        o_t_1 is of shape (batch_size, hidden_dim) - (B, H)
        """

        # Compute the initial hidden
        # h_0 = tanh( W_h @ mean(x) + b_h )
        if h_t_1 is None:
            # h_t_1 = torch.zeros(x.shape[0], self.config['hidden_dim'], device=self.config['DEVICE'])
            ctx_mean = (torch.einsum('...l, ...ld -> ...d', feature_mask.squeeze(), x).squeeze() /
                        torch.sum(feature_mask.squeeze(), dim=-1).unsqueeze(-1))
            h_t_1 = torch.tanh(self.parser['W_2h'](ctx_mean))  # (B, H)

        # Compute the attention weights and context vector
        # e_t = v_a^T tanh( U_a @ x + W_a @ h_t_1 )
        # try:
        e_t = torch.tanh(self.parser['U_a'](x) + self.parser['W_a'](h_t_1).unsqueeze(-2))  # (B, L, A)
        e_t = self.parser['nu_a'](e_t).squeeze() + torch.where(feature_mask == 0, -torch.inf, 1).squeeze()  # (B, L)
        alpha_t = torch.softmax(e_t, dim=-1)  # (B, L)
        ct = torch.einsum('...l, ...ld -> ...d', alpha_t, x.squeeze())  # (B, D) or (Beam_width, B, D)

        # Forward Pass through GRU
        # Embed the target sentence
        ey_t_1 = self.embedder(y).squeeze()  # (B, E)

        # Compute the z gate
        # z_t = sigmoid( W_yz @ e_t_1 + U_hz @ h_t_1 + C_cz @ c_t )
        zt = torch.sigmoid(self.parser['W_yz'](ey_t_1) + self.parser['U_hz'](h_t_1))  # (B, H)

        # Compute the reset gate
        # r_t = sigmoid( W_yr @ e_t_1 + U_hr @ h_t_1 + C_cr @ c_t )
        rt = torch.sigmoid(self.parser['W_yr'](ey_t_1) + self.parser['U_hr'](h_t_1))  # (B, H)

        # Compute the candidate hidden state
        # h_tilde = tanh( W_yh @ e_t_1 + U_rh @ (r_t * h_t_1) + C_cz @ c_t )
        rt_ht = rt * h_t_1  # (B, H)
        ht_tilde = torch.tanh(
            self.parser['W_yh'](ey_t_1) + self.parser['U_rh'](rt_ht) + self.parser['C_cz'](ct))  # (B, H)

        # Compute the new hidden state
        h_t = (1 - zt) * h_t_1 + zt * ht_tilde  # (B, H)

        # Compute the output
        # o_t = W_o @ ( W_c @ c_t + W_h @ h_t + e_t_1 )
        logit_ctx = self.parser['W_c'](ct)  # (B, E)
        logit_ht = self.parser['W_h'](h_t)  # (B, E)
        logit_ey = self.parser['W_yo'](ey_t_1)  # (B, E)

        logit = logit_ctx + logit_ht + logit_ey  # (B, E)

        # reshaped logit and max out layer
        shape = list(logit.shape)
        shape = tuple(shape[:-1] + [shape[-1] // 2, 2])
        logit = torch.max(logit.reshape(shape), dim=-1)[0]

        o_t = self.parser['W_o'](logit)  # (B, V)
        # o_t = self.parser['W_o'](self.parser['W_c'](ct) + self.parser['W_h'](h_t) + ey_t_1)  # (B, V)

        return o_t, h_t

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

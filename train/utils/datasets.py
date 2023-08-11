import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

def collate_fn(batch):
    # Separate inputs and labels
    images, image_mask, labels, seq_len, labels_mask = zip(*batch)

    # Pad sequences
    max_h = max([image.shape[1] for image in images])
    max_w = max([image.shape[2] for image in images])

    # Pad image and image_mask
    images = list(images)
    image_mask = list(image_mask)
    for i in range(len(images)):
        padding = (0, max_w-images[i].shape[2], 0, max_h-images[i].shape[1])
        images[i] = torch.nn.functional.pad(images[i], padding, "constant", 0)
        image_mask[i] = torch.nn.functional.pad(image_mask[i], padding, "constant", 0)


    images = torch.stack(images)
    image_mask = torch.stack(image_mask)


    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    labels_mask = pad_sequence(labels_mask, batch_first=True, padding_value=0)
    seq_lens = torch.tensor(seq_len)
    return images, image_mask, labels, seq_lens, labels_mask

def get_vocabulary(csv_loc):
    """
    :param csv_loc: location of the csv file
    :return:
    """
    # Read the csv file
    with open(csv_loc, 'r') as f:
        vocabulary = f.read().strip().split('\n')

    vocabulary.sort()
    vocabulary = ['<SOS>', '<EOS>'] + vocabulary

    return vocabulary

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, vocab_loc, device, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.device = device

        self.vocab = get_vocabulary(vocab_loc)
        self.word_to_index = {word: i for i, word in enumerate(self.vocab)}
        self.index_to_word = {i: word for i, word in enumerate(self.vocab)}

        # assert self.vocab[0] == '<SOS>' and self.vocab[0] == '<EOS>', 'The third and fourth element of the vocab must be <SOS> and <EOS> respectively'

    def __getitem__(self, index):
        # Load Image in grayscale, and transform it
        image = Image.open(self.image_paths[index]).convert('L')
        if self.transform is not None:
            image = self.transform(image)

        # Tokenize the label
        sentence = self.labels[index]
        if type(sentence) != str:
            sentence = str(sentence)

        tokenized_sentences = self.tokenize(sentence)

        image = image.to(self.device)
        image_mask = torch.ones_like(image).to(self.device)
        tensor_sentence = torch.tensor(tokenized_sentences).to(self.device)
        anno_mask = torch.ones_like(tensor_sentence).to(self.device)
        seq_len = torch.tensor(len(tensor_sentence)).to(self.device)

        return image, image_mask, tensor_sentence, seq_len, anno_mask

    def __len__(self):
        return len(self.image_paths)

    def tokenize(self, sentence):
        ret = []
        for word in sentence.split():
            if word not in self.word_to_index:
                exit('Word not in vocabulary')
            ret.append(self.word_to_index[word])
        ret.append(self.word_to_index["<EOS>"])
        return  ret

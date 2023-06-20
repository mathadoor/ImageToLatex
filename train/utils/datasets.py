import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

def collate_fn(batch):
    # Separate inputs and labels
    inputs, labels = zip(*batch)

    # Pad sequences
    inputs = pad_sequence(inputs, batch_first=True)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)

    return inputs, labels

def get_vocabulary(csv_loc):
    """
    :param csv_loc: location of the csv file
    :return:
    """
    # Read the csv file
    with open(csv_loc, 'r') as f:
        vocabulary = f.read().split('\n')

    vocabulary.sort()
    vocabulary = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + vocabulary

    return vocabulary

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, vocab_loc, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        self.vocab = get_vocabulary(vocab_loc)
        self.word_to_index = {word: i for i, word in enumerate(self.vocab)}
        self.index_to_word = {i: word for i, word in enumerate(self.vocab)}

        assert self.vocab[2] == '<SOS>' and self.vocab[3] == '<EOS>', 'The third and fourth element of the vocab must be <SOS> and <EOS> respectively'

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
        tensor_sentence = torch.tensor(tokenized_sentences)

        return image, tensor_sentence

    def __len__(self):
        return len(self.image_paths)

    def tokenize(self, sentence):
        ret = [self.word_to_index["<SOS>"]]
        for word in sentence.split():
            if word not in self.word_to_index:
                ret.append(self.word_to_index["<UNK>"])
                continue
            ret.append(self.word_to_index[word])
        ret.append(self.word_to_index["<EOS>"])
        return  ret

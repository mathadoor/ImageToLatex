from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, vocab, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.vocab = vocab
        self.word_to_index = {word: i for i, word in enumerate(self.vocab)}
        self.index_to_word = {i: word for i, word in enumerate(self.vocab)}

    def __getitem__(self, index):
        # load image as ndarray type (H x W x C) in grey scale format
        image = Image.open(self.image_paths[index]).convert('L')

        # apply image transformation
        if self.transform is not None:
            image = self.transform(image)

        return image, self.labels[index]

    def __len__(self):
        return len(self.image_paths)



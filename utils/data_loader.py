from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Going to have to import the inkML files instead to accomodate the data.
# I think I might as well create different classes pertaining to stroke data. But that is not important at this point

# Custom Class to load the data into dataloader
class StrokeData(Dataset):

    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):

        # Load the data from the path
        data = np.load(self.path[idx])

        # Get the label from the path
        label = self.path[idx].split('/')[-1].split('.')[0]

        # Transform the data
        if self.transform:
            data = self.transform(data)

        return data, label
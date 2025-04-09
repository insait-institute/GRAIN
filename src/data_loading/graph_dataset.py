from torch.utils.data import Dataset, DataLoader

class GraphDataset(Dataset):
    def __init__(self, args):
        self.args = args

    def __len__(self):
        return NotImplementedError
    
    def __getitem__(self, idx):
        return NotImplementedError 
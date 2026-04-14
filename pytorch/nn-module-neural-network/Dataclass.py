from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self,X,Y):
        self.x = X
        self.y = Y
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
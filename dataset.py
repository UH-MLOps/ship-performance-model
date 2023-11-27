import torch
from torch.utils.data import Dataset

# Ensure that FuelMassFlow is stored in the last column of the csv. 
# Using example_dataset, the last column 'FuelMassFlow' is at '9'
class dds(Dataset):
    def __init__(self, df):
        x = df.iloc[:, :9].values
        y = df.iloc[:, 9].values    

        self.x_tensor = torch.tensor(x, dtype = torch.float32)
        self.y_tensor = torch.tensor(y, dtype = torch.float32).reshape(-1,1)

    def __len__(self):
        return len(self.y_tensor)

    def __getitem__(self,idx):
        return self.x_tensor[idx], self.y_tensor[idx]
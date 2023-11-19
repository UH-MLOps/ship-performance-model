import torch
from torch.utils.data import Dataset

# For this part, making sure that FuelMassFlow is stored in the 15th column in the csv. 
# Otherwise change the number '14' and ':14' accordingly
class dds(Dataset):
    def __init__(self, df):
        x = df.iloc[:, :14].values
        y = df.iloc[:, 14].values    

        self.x_tensor = torch.tensor(x, dtype = torch.float32)
        self.y_tensor = torch.tensor(y, dtype = torch.float32).reshape(-1,1)

    def __len__(self):
        return len(self.y_tensor)

    def __getitem__(self,idx):
        return self.x_tensor[idx], self.y_tensor[idx]
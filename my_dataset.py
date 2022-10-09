import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import model_selection


class MyDataset(Dataset):
    def __init__(self, is_train=True):
        self.X_SIZE = 300
        self.is_train = is_train
        self.csv_file_location = "data/data.csv"
        self.df = pd.read_csv(self.csv_file_location)
        train, test = model_selection.train_test_split(self.df, test_size=0.2)
        self.df = train
        if not self.is_train:
            self.df = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.df.iloc[idx,0:self.X_SIZE]
        y = self.df.iloc[idx,self.X_SIZE]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def get_x(self):
        return self.df[self.df.columns[0:self.X_SIZE]].values

    def get_y(self):
        return self.df[self.df.columns[self.X_SIZE]].values

if __name__ == "__main__":
    cid = MyDataset()
    dataloader = DataLoader(cid, batch_size=1, shuffle=True)
    for x, soc in dataloader:
        print(x)
        print(x.shape[1])
        exit(0)

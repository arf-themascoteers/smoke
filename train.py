import torch
import torch.nn.functional as F
from my_dataset import MyDataset
import torch.nn as nn
from torch.utils.data import DataLoader
from my_machine import MyMachine

def train(device):
    batch_size = 10000
    cid = MyDataset(is_train=True)
    dataloader = DataLoader(cid, batch_size=batch_size, shuffle=True)
    model = MyMachine()
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-3)
    criterion = torch.nn.MSELoss(reduction='mean')
    num_epochs = 1000
    n_batches = int(len(cid)/batch_size) + 1
    batch_number = 0
    loss = None

    for epoch in range(num_epochs):
        batch_number = 0
        for (x, y) in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            y_hat = y_hat.reshape(-1)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            batch_number += 1
            print(f'Epoch:{epoch + 1} (of {num_epochs}), Batch: {batch_number} of {n_batches}, Loss:{loss.item():.4f}')

    print("Train done")
    torch.save(model, 'models/soc.h5')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(device)
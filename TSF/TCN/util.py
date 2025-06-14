import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):

    """
        Converts multivariate time series from a pandas DataFrame into sliding windows.
    
        Args:
            df (pd.DataFrame): Time series data, shape (T, D+1), where first column is timestamp
            input_len   (int): Number of time steps in input sequence
            output_len  (int): Number of future steps to forecast
    """

    def __init__(
        self, 
        df         : pd.DataFrame, 
        input_len  : int, 
        output_len : int
    ):

        self.data = df.iloc[:, 1:].values.astype(np.float32) # drop timestamp
        self.input_len  = input_len
        self.output_len = output_len
        self.seq_len    = input_len + output_len

    def __len__(self):

        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):

        seq = self.data[idx: idx+self.seq_len]
        x   = seq[:self.input_length]          
        y   = seq[self.input_length:]
        return torch.from_numpy(x), torch.from_numpy(y)


def train_one_epoch(model, dataloader, optimizer, criterion, device):

    model.train()
    total_loss = 0.0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad() # reset the gradient to zero 
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step() 
        total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset) # return avg of loss


def evaluate(model, dataloader, criterion, device):
    
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)



import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SalaryDataset(Dataset):
    def __init__(self, cat_data, text_data, y=None):
        self.cat_features = torch.tensor(cat_data.values, dtype=torch.long)
        self.text_features = torch.tensor(text_data, dtype=torch.long)

        # If target exists (training), store it. Otherwise (prediction), it's none.
        self.y = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.text_features)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.cat_features[idx], self.text_features[idx], self.y[idx]
        return self.cat_features[idx], self.text_features[idx]
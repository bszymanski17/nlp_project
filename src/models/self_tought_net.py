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

class SalaryPredictionModel(nn.Module):
    def __init__(self, cat_vocab_sizes, text_vocab_size, config):
        super(SalaryPredictionModel, self).__init__()
        
        emb_dim_cat = config['model']['emb_dim_cat']
        emb_dim_text = config['model']['emb_dim_text']

        # Categorical Embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=emb_dim_cat, padding_idx=0) 
            for size in cat_vocab_sizes
        ])

        # Text Embedding
        self.text_embedding = nn.Embedding(
            num_embeddings=text_vocab_size, 
            embedding_dim=emb_dim_text, 
            padding_idx=0
        )

        input_dim = len(cat_vocab_sizes) * emb_dim_cat + emb_dim_text

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.out = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x_cat, x_text):
        # Process categorical features through embeddings
        cat_embs = [layer(x_cat[:, i]) for i, layer in enumerate(self.cat_embeddings)]
        cat_embs = torch.cat(cat_embs, dim=1)

        # Process text through embedding and calculate mean (Global Average Pooling)
        text_emb = self.text_embedding(x_text)
        text_emb_mean = torch.mean(text_emb, dim=1)

        # Concatenate all features
        combined = torch.cat([cat_embs, text_emb_mean], dim=1)

        # Pass through dense layers
        x = self.relu(self.fc1(combined))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        
        return self.out(x)
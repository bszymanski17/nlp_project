import torch
import torch.nn as nn

class SalaryPredictionModel(nn.Module):
    def __init__(self, cat_vocab_sizes, text_vocab_size, config):
        super(SalaryPredictionModel, self).__init__()
        
        emb_dim_cat = config['model']['emb_dim_cat']
        emb_dim_text = config['model']['emb_dim_text']
        layers = config['model']['layers_size']

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

        self.fc1 = nn.Linear(input_dim, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], layers[2])
        self.out = nn.Linear(layers[2], 1)
        self.relu = nn.ReLU()

    def forward(self, x_cat, x_text):
        cat_embs = [layer(x_cat[:, i]) for i, layer in enumerate(self.cat_embeddings)]
        cat_embs = torch.cat(cat_embs, dim=1)

        text_emb = self.text_embedding(x_text)
        text_emb_mean = torch.mean(text_emb, dim=1)

        # Concatenate all features
        combined = torch.cat([cat_embs, text_emb_mean], dim=1)

        x = self.relu(self.fc1(combined))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        
        return self.out(x)
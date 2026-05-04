import pandas as pd
import torch
import nltk
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

from src.utils import load_config, get_logger, ensure_dir
from src.pipeline.preprocessing import SalaryPreprocessor
from src.models.self_tought_net import SalaryPredictionModel
from src.models.dataset import SalaryDataset
from src.pipeline.operations import train_model

def main():
    logger = get_logger("Main")
    config = load_config("config.yaml")
    
    nltk.download('punkt', quiet=True)
    
    ensure_dir(config['paths']['artifacts_dir'])

    device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")

    logger.info(f"Loading data from {config['data']['path']}...")
    df = pd.read_csv(config['data']['path']) 
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=config['training']['random_state'])

    preprocessor = SalaryPreprocessor(config)
    preprocessor.fit(train_df)
    
    x_cat_train, x_text_train = preprocessor.transform(train_df)
    x_cat_val, x_text_val = preprocessor.transform(val_df)
    
    preprocessor.save(config['paths']['preprocessor_save'])

    # Prepare DataLoaders
    train_dataset = SalaryDataset(x_cat_train, x_text_train, np.log1p(train_df[config['data']['target']]))
    val_dataset = SalaryDataset(x_cat_val, x_text_val, np.log1p(val_df[config['data']['target']]))
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])

    model = SalaryPredictionModel(
        cat_vocab_sizes=preprocessor.vocab_size_list,
        text_vocab_size=len(preprocessor.word_to_id),
        config=config
    ).to(device)

    logger.info("Starting training...")
    trained_model = train_model(model, train_loader, val_loader, config, device)

    logger.info("Training completed successfully.")

if __name__ == "__main__":
    main()
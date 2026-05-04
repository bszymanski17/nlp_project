import torch
import torch.nn as nn
import mlflow
import numpy as np
from tqdm import tqdm
from src.utils import get_logger

logger = get_logger("Operations")

def train_model(model, train_loader, val_loader, config, device):
    """
    Main training loop.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = config['training']['epochs']
    patience = config['training']['patience']
    best_val_loss = float('inf')
    counter = 0

    logger.info(f"Starting training for {epochs} epochs on device: {device}")

    with mlflow.start_run():
        mlflow.log_params(config['training'])
        mlflow.log_params(config['model'])

        for epoch in range(epochs):
            # TRAINING
            model.train()
            train_losses = []
            
            for x_cat, x_text, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                x_cat, x_text, y = x_cat.to(device), x_text.to(device), y.to(device)
                
                optimizer.zero_grad()
                outputs = model(x_cat, x_text)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # VALIDATION
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x_cat, x_text, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    x_cat, x_text, y = x_cat.to(device), x_text.to(device), y.to(device)
                    outputs = model(x_cat, x_text)
                    loss = criterion(outputs, y)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

            # Early Stopping 
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), config['paths']['model_save'])
                logger.info(f"New best model saved with Val Loss: {best_val_loss:.4f}")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    logger.warning(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        mlflow.log_artifact(config['paths']['model_save'])
        logger.info("Training complete. Best model and metrics logged to MLflow.")

    return model
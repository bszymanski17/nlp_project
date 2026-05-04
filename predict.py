import torch
import pandas as pd
import numpy as np
import sys
from src.utils import load_config, get_logger
from src.pipeline.preprocessing import SalaryPreprocessor
from src.models.self_tought_net import SalaryDataset, SalaryPredictionModel

logger = get_logger("Inference")

def run_batch_prediction():
    try:
        config = load_config("config.yaml")
        device = torch.device("cpu") 

        logger.info("Loading pre-fitted preprocessor and model...")
        preprocessor = SalaryPreprocessor.load(config['paths']['preprocessor_save'])
        
        data_path = config['data']['path']
        logger.info(f"Loading data from {data_path} for prediction...")
        df = pd.read_csv(data_path)

        logger.info(f"Transforming {len(df)} rows...")
        x_cat, x_text = preprocessor.transform(df)
        
        x_cat_tensor = torch.tensor(x_cat.values, dtype=torch.long).to(device)
        x_text_tensor = torch.tensor(x_text, dtype=torch.long).to(device)

        model = SalaryPredictionModel(
            cat_vocab_sizes=preprocessor.vocab_size_list,
            text_vocab_size=len(preprocessor.word_to_id),
            config=config
        ).to(device)
        
        model.load_state_dict(torch.load(config['paths']['model_save'], map_location=device))
        model.eval()

        logger.info("Calculating predictions...")
        with torch.no_grad():
            # Model outputs log-salary
            predictions_log = model(x_cat_tensor, x_text_tensor)
        

        logger.info("Applying exponential transformation (expm1)...")
        predictions_real = np.expm1(predictions_log.numpy()).flatten()

        # Save results
        df['Predicted_Salary'] = predictions_real
        output_path = config['paths']['prediction_results']
        df.to_csv(output_path, index=False)
        
        logger.info(f"DONE! Results saved to: {output_path}")
        logger.info(f"Average predicted salary: £{predictions_real.mean():,.2f}")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_batch_prediction()
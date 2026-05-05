import torch
import pandas as pd
import numpy as np
import sys
import mlflow
from src.utils import load_config, get_logger
from src.processing.preprocessing import SalaryPreprocessor
from src.models.self_tought_net import SalaryPredictionModel

logger = get_logger("Prediction")

def run_batch_prediction():
    try:
        config = load_config("config.yaml")
        device = torch.device("cpu") 

        mlflow.set_tracking_uri(config['paths']['tracking_uri'])
                
        run_id = config['mlflow']['run_id']
        model_name = config['mlflow'].get('model_name', 'model')
        model_uri = f"runs:/{run_id}/{model_name}"


        with mlflow.start_run(run_name="Predictions"):

            logger.info(f"Loading pre-fitted preprocessor and model {run_id}/{model_name} ...")
            preprocessor = SalaryPreprocessor.load(config['paths']['preprocessor_save'])
            model = mlflow.pytorch.load_model(model_uri)
            model.eval()
            
            data_path = config['paths']['input_data_pred']
            logger.info(f"Loading data from {data_path} for prediction...")
            df = pd.read_csv(data_path)

            x_cat, x_text = preprocessor.transform(df)
            
            x_cat_tensor = torch.tensor(x_cat.values, dtype=torch.long).to(device)
            x_text_tensor = torch.tensor(x_text, dtype=torch.long).to(device)

            logger.info("Calculating predictions...")

            with torch.no_grad():
                predictions_log = model(x_cat_tensor, x_text_tensor)
            
            predictions_real = np.expm1(predictions_log.numpy()).flatten()
            df['Predicted_Salary'] = predictions_real


            output_path = config['paths']['prediction_results']
            df.to_csv(output_path, index=False)

            mlflow.log_param("model_file", config['mlflow'].get('run_id', 'unknown'))
            mlflow.log_artifact(output_path)
            
            logger.info(f"Results saved to: {output_path}")
            logger.info(f"Average predicted salary: {predictions_real.mean():,.2f}")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_batch_prediction()
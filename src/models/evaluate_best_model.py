import pandas as pd
import numpy as np
import json
import logging
import joblib
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

def setup_logging():
    """Configure logging."""
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    return logging.getLogger(__name__)

def load_model_and_data(models_path, data_path):
    """Load trained model and test data."""
    logger = logging.getLogger(__name__)
    
    # Load model
    model_path = os.path.join(models_path, 'final_trained_model.pkl')
    model = joblib.load(model_path)
    logger.info("Model loaded")
    
    # Load test data
    X_test = pd.read_csv(os.path.join(data_path, 'processed_data', 'X_test_scaled.csv'))
    y_test = pd.read_csv(os.path.join(data_path, 'processed_data', 'y_test.csv')).iloc[:, 0]
    logger.info(" Test data loaded")
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model and calculate metrics."""
    logger = logging.getLogger(__name__)
    logger.info("Evaluating model...")
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2_score': float(r2),
        'evaluation_date': datetime.now().isoformat()
    }
    
    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    
    return predictions, metrics

def save_predictions(X_test, y_test, predictions, data_path):
    """Save predictions dataset."""
    logger = logging.getLogger(__name__)
    
    # Create predictions dataframe
    predictions_df = X_test.copy()
    predictions_df['actual'] = y_test.values
    predictions_df['predicted'] = predictions
    predictions_df['error'] = y_test.values - predictions
    
    # Save to data folder
    predictions_path = os.path.join(data_path, 'model_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    
    logger.info(f" Predictions saved to: {predictions_path}")

def save_scores(metrics, metrics_path):
    """Save scores.json."""
    logger = logging.getLogger(__name__)
    
    os.makedirs(metrics_path, exist_ok=True)
    
    scores_path = os.path.join(metrics_path, 'scores.json')
    with open(scores_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Scores saved to: {scores_path}")

def main(data_path="./data", models_path="./models", metrics_path="./metrics"):
    """Main evaluation function."""
    logger = setup_logging()
    logger.info(" Starting model evaluation...")
    
    try:
        # Load model and data
        model, X_test, y_test = load_model_and_data(models_path, data_path)
        
        # Evaluate model
        predictions, metrics = evaluate_model(model, X_test, y_test)
        
        # Save predictions dataset
        save_predictions(X_test, y_test, predictions, data_path)
        
        # Save scores
        save_scores(metrics, metrics_path)
        
        logger.info(" Evaluation completed!")
        
    except Exception as e:
        logger.error(f" Error: {e}")
        raise

if __name__ == '__main__':
    main()
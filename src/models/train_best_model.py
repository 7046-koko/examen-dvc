import pandas as pd
import pickle
import logging
import joblib
import os

def setup_logging():
    """Configure logging."""
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    return logging.getLogger(__name__)

def load_best_model_from_pkl(models_path):
    """Load the best model from pkl file."""
    logger = logging.getLogger(__name__)
    
    best_model_file = os.path.join(models_path, 'best_model.pkl')
    
    with open(best_model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    best_model = model_data['model']
    best_params = model_data['best_parameters']
    
    logger.info(f" Best model loaded from: {best_model_file}")
    logger.info(f"Model parameters: {best_params}")
    
    return best_model

def load_training_data(data_path):
    """Load training data."""
    logger = logging.getLogger(__name__)
    
    X_train = pd.read_csv(os.path.join(data_path, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv')).iloc[:, 0]
    
    logger.info(f"Training data loaded - X_train: {X_train.shape}, y_train: {y_train.shape}")
    return X_train, y_train

def train_model_from_best(best_model, X_train, y_train):
    """Train the best model on training data."""
    logger = logging.getLogger(__name__)
    logger.info(" Training model from best parameters...")
    
    # Train the model (it already has the best parameters)
    best_model.fit(X_train, y_train)
    
    logger.info(" Model training completed")
    return best_model

def save_final_trained_model(trained_model, models_path):
    """Save the final trained model."""
    logger = logging.getLogger(__name__)
    
    os.makedirs(models_path, exist_ok=True)
    
    final_model_path = os.path.join(models_path, 'final_trained_model.pkl')
    joblib.dump(trained_model, final_model_path)
    
    logger.info(f"Final trained model saved to: {final_model_path}")

def main(data_path="./data/processed_data", models_path="./models"):
    """Main training function using best model from pkl."""
    logger = setup_logging()
    logger.info("Starting training from best model in pkl...")
    
    try:
        # Load best model from pkl
        best_model = load_best_model_from_pkl(models_path)
        
        # Load training data
        X_train, y_train = load_training_data(data_path)
        
        # Train the model
        final_trained_model = train_model_from_best(best_model, X_train, y_train)
        
        # Save final trained model
        save_final_trained_model(final_trained_model, models_path)
        
        logger.info(" Training from pkl completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()
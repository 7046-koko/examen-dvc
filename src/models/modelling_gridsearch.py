import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import os

def setup_logging():
    """Configure logging."""
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    return logging.getLogger(__name__)

def load_data(data_path):
    """Load preprocessed training data."""
    logger = logging.getLogger(__name__)
    
    X_train_scaled = pd.read_csv(os.path.join(data_path, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv')).iloc[:, 0]
    
    logger.info(f"Data loaded - X_train_scaled: {X_train_scaled.shape}, y_train: {y_train.shape}")
    return X_train_scaled, y_train

def train_model(X_train_scaled, y_train):
    """Train RandomForest with GridSearch."""
    logger = logging.getLogger(__name__)
    logger.info("Starting GridSearch...")
    
    # Define model and parameters
    model = RandomForestRegressor(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # GridSearch
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Train
    grid_search.fit(X_train_scaled, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def save_best_model(best_model, best_params, output_path):
    """Save only the best model with parameters."""
    logger = logging.getLogger(__name__)
    
    # Create models directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save only the best model and parameters
    model_data = {
        'model': best_model,
        'best_parameters': best_params
    }
    
    model_path = os.path.join(output_path, 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f" Best model saved to: {model_path}")
    logger.info(f" Parameters: {best_params}")

def main(data_path="./data/processed_data", output_path="./models"):
    """Main function."""
    logger = setup_logging()
    logger.info(" Training RandomForest to find best parameters...")
    
    try:
        # Load data
        X_train_scaled, y_train = load_data(data_path)
        
        # Train and get best model
        best_model, best_params = train_model(X_train_scaled, y_train)
        
        # Save best model
        save_best_model(best_model, best_params, output_path)
        
        logger.info(" Training completed!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise

if __name__ == '__main__':
    main()
import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from .model import StockDataset, ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_path, sequence_length=10):
    """
    Load and preprocess the data for training.
    
    Args:
        data_path (str): Path to the processed data file
        sequence_length (int): Length of input sequences
        
    Returns:
        tuple: Processed features and targets
    """
    # Load the processed data
    df = pd.read_csv(data_path)
    
    # Separate features and target
    target_col = 'Target'  # Binary classification target
    feature_cols = [col for col in df.columns if col not in [target_col, 'Date', 'Target_Return']]
    
    # Extract features and target
    features = df[feature_cols].values
    targets = df[target_col].values
    
    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Prepare sequences
    X, y = [], []
    for i in range(len(features_scaled) - sequence_length):
        X.append(features_scaled[i:(i + sequence_length)])
        y.append(targets[i + sequence_length])
    
    return np.array(X), np.array(y)

class EnsemblePredictor:
    def __init__(self, models, device):
        self.models = models
        self.device = device

    def predict(self, features):
        predictions = []
        with torch.no_grad():
            features = torch.FloatTensor(features).to(self.device)
            for model in self.models:
                model.eval()
                pred = model(features)
                predictions.append(pred)
        
        # Average predictions
        ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        return ensemble_pred

def train_ensemble(data_dir, symbol, num_models=3, model_params=None):
    """
    Train an ensemble of models for better prediction.
    
    Args:
        data_dir (str): Directory containing the processed data
        symbol (str): Stock symbol to train on
        num_models (int): Number of models in the ensemble
        model_params (dict): Model parameters
    """
    if model_params is None:
        model_params = {
            'sequence_length': 10,
            'hidden_dim': 128,
            'num_layers': 2,
            'batch_size': 32,
            'train_split': 0.8,
            'val_split': 0.1,
            'epochs': 100
        }
    
    # Load and preprocess data
    data_path = Path(data_dir) / 'processed' / f'{symbol}_processed.csv'
    logger.info(f'\nLoading data for {symbol} from {data_path}')
    X, y = load_and_preprocess_data(data_path, model_params['sequence_length'])
    
    logger.info(f'Dataset shape - Features: {X.shape}, Targets: {y.shape}')
    
    # Create dataset
    dataset = StockDataset(X, y)
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(model_params['train_split'] * total_size)
    val_size = int(model_params['val_split'] * total_size)
    test_size = total_size - train_size - val_size
    
    logger.info(f'Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}')
    
    # Train multiple models
    models = []
    for i in range(num_models):
        logger.info(f'\nTraining model {i+1}/{num_models}')
        
        # Create new train/val/test splits for each model
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(i)  # Different seed for each model
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=model_params['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=model_params['batch_size']
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=model_params['batch_size']
        )
        
        # Initialize and train model
        input_dim = X.shape[2]  # Number of features
        trainer = ModelTrainer(
            input_dim,
            model_params['sequence_length'],
            model_params['hidden_dim'],
            model_params['num_layers']
        )
        
        # Train the model
        trainer.train(train_loader, val_loader, model_params['epochs'])
        
        # Evaluate on test set
        test_loss, test_accuracy = evaluate_model(trainer.model, test_loader, trainer.criterion, trainer.device)
        logger.info(f'Test Results for Model {i+1}:')
        logger.info(f'Loss: {test_loss:.4f}')
        logger.info(f'Accuracy: {test_accuracy:.2f}%')
        
        # Save individual model
        model_path = trainer.models_dir / f'{symbol}_model_{i+1}.pth'
        torch.save(trainer.model.state_dict(), model_path)
        logger.info(f'Model saved as {model_path.name}')
        
        models.append(trainer.model)
    
    # Create and save ensemble
    ensemble = EnsemblePredictor(models, trainer.device)
    
    # Evaluate ensemble on test set
    ensemble_accuracy = evaluate_ensemble(ensemble, test_loader, trainer.device)
    logger.info(f'\nEnsemble Test Accuracy: {ensemble_accuracy:.2f}%')
    
    return ensemble

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate a single model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            predicted = (outputs.data > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate_ensemble(ensemble, data_loader, device):
    """Evaluate the ensemble model"""
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            outputs = ensemble.predict(features)
            predicted = (outputs.data > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def main():
    """Main function to train models for all symbols"""
    data_dir = Path(__file__).parent.parent.parent / 'data'
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    for symbol in symbols:
        logger.info(f'\n{"="*50}')
        logger.info(f'Training ensemble for {symbol}')
        logger.info(f'{"="*50}')
        
        ensemble = train_ensemble(data_dir, symbol, num_models=3)

if __name__ == "__main__":
    main()

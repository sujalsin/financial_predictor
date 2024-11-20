import torch
import logging
from pathlib import Path
from src.models.train_model import load_and_preprocess_data, StockDataset, EnsemblePredictor
from torch.utils.data import DataLoader
import pandas as pd
from src.models.model import LSTMPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models(model_dir, symbol, device):
    """Load the ensemble models for a given symbol"""
    models = []
    for i in range(1, 4):  # Load 3 ensemble models
        model_path = model_dir / f'{symbol}_model_{i}.pth'
        model = LSTMPredictor(input_dim=24, hidden_dim=128, num_layers=2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    return models

def analyze_stock(symbol, data_dir, model_dir, device):
    """Analyze performance for a single stock"""
    # Load and preprocess data
    data_path = data_dir / 'processed' / f'{symbol}_processed.csv'
    X, y = load_and_preprocess_data(data_path)
    
    # Create dataset and dataloader
    dataset = StockDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load models
    models = load_models(model_dir, symbol, device)
    ensemble = EnsemblePredictor(models, device)
    
    # Evaluate ensemble
    correct = 0
    total = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            outputs = ensemble.predict(features)
            predicted = (outputs.data > 0.5).float()
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    accuracy = 100 * correct / total
    return {
        'symbol': symbol,
        'accuracy': accuracy,
        'total_predictions': total,
        'correct_predictions': correct
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_dir = Path(__file__).parent.parent.parent
    data_dir = project_dir / 'data'
    model_dir = project_dir / 'models'
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    results = []
    
    logger.info(f'\nAnalyzing model performance on {device}...\n')
    logger.info('='*50)
    
    for symbol in symbols:
        logger.info(f'\nAnalyzing {symbol}...')
        result = analyze_stock(symbol, data_dir, model_dir, device)
        results.append(result)
        logger.info(f'Accuracy: {result["accuracy"]:.2f}%')
        logger.info(f'Total predictions: {result["total_predictions"]}')
        logger.info(f'Correct predictions: {result["correct_predictions"]}')
        logger.info('-'*50)
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    logger.info('\nSummary of Results:')
    logger.info('\n' + str(df))
    
    # Calculate average accuracy
    avg_accuracy = df['accuracy'].mean()
    logger.info(f'\nAverage accuracy across all stocks: {avg_accuracy:.2f}%')

if __name__ == '__main__':
    main()

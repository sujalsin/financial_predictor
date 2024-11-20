# Financial Market Prediction System

An advanced machine learning system for predicting stock market trends using deep learning techniques and ensemble modeling.

## Project Overview

- **Type**: Machine Learning Stock Price Prediction Project
- **Primary Technology Stack**: Python, PyTorch, Pandas, Scikit-learn
- **Goal**: Develop a predictive analytics system for financial market trends using advanced deep learning techniques

## Model Architecture

### Key Components

1. **Attention Mechanism**
   - Helps focus on important time steps in sequence data
   - Implemented in `AttentionLayer` class

2. **Residual LSTM Blocks**
   - Improves gradient flow
   - Helps mitigate vanishing gradient problem
   - Implemented in `ResidualLSTMBlock` class

3. **Enhanced Fully Connected Layers**
   - Layer normalization
   - Dropout for regularization
   - Multi-layer architecture with residual connections

### Training Process
- L2 Regularization (weight decay = 0.01)
- Gradient Clipping (max norm = 1.0)
- Enhanced Early Stopping Mechanism
- Comprehensive Logging and Metrics Tracking

### Ensemble Strategy
- Multiple models per stock symbol
- Different random seeds for data splitting
- Model averaging for predictions

## Performance Results

### Overall Performance
- **Average Accuracy**: 79.29%
- **Prediction Window**: 10-day sequences

### Stock-Specific Performance
| Stock | Accuracy | Total Predictions | Correct Predictions |
|-------|----------|-------------------|-------------------|
| AMZN  | 84.90%   | 437              | 371               |
| MSFT  | 81.69%   | 437              | 357               |
| AAPL  | 76.43%   | 437              | 334               |
| GOOGL | 74.14%   | 437              | 324               |

## Technical Details

### Model Hyperparameters
- Hidden Dimensions: 128
- LSTM Layers: 2
- Batch Size: 32
- Training Split: 80%
- Validation Split: 10%
- Test Split: 10%
- Epochs: 100 (with early stopping)

### Features
- Technical indicators
- Price-derived features
- 10-day sequence windows
- Binary classification (price movement prediction)

## Project Structure
```
financial_predictor/
├── data/
│   ├── raw/           # Original stock data
│   └── processed/     # Engineered feature datasets
├── src/
│   ├── data/          # Data collection scripts
│   ├── features/      # Feature engineering
│   └── models/
│       ├── model.py   # Enhanced LSTM model architecture
│       └── train_model.py  # Ensemble training pipeline
├── models/            # Saved model checkpoints
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Dependencies
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- yfinance (for data collection)

## Environment Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Data Collection:
   ```python
   python -m src.data.collect_data
   ```

2. Training:
   ```python
   python -m src.models.train_model
   ```

3. Analysis:
   ```python
   python -m src.models.analyze_results
   ```

## Future Improvements

1. Feature Engineering
   - Implement more sophisticated technical indicators
   - Add sentiment analysis features
   - Explore alternative data sources

2. Model Enhancements
   - Experiment with different ensemble techniques
   - Implement cross-validation
   - Add model interpretation tools

3. System Improvements
   - Real-time prediction pipeline
   - Web interface for predictions
   - Automated model retraining

## Known Limitations
- Binary classification of price movement
- Limited to specific stock symbols
- Potential overfitting risks
- Computational complexity of ensemble approach

## Security Considerations
- No sensitive credentials stored in code
- Public API data sources
- Modular design allows easy credential management

## License
MIT License

## Contributors
- Initial development by the Codeium team

## Acknowledgments
Special thanks to the open-source community and the creators of the libraries used in this project.

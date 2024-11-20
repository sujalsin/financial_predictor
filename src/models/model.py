import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets).reshape(-1, 1)  # Reshape targets to match model output
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended_output = torch.sum(attention_weights * lstm_output, dim=1)
        return attended_output, attention_weights

class ResidualLSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(ResidualLSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            output, (hn, cn) = self.lstm(x)
        else:
            output, (hn, cn) = self.lstm(x, (h0, c0))
        output = self.dropout(output)
        if x.size(-1) == output.size(-1):  # If dimensions match
            output = output + x
        output = self.norm(output)
        return output, hn, cn

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Residual LSTM layers
        self.lstm_layers = nn.ModuleList([
            ResidualLSTMBlock(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                dropout
            ) for i in range(num_layers)
        ])
        
        # Attention layer
        self.attention = AttentionLayer(hidden_dim)
        
        # Improved FC layers with residual connections
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        
        # Output layer
        self.output = nn.Linear(hidden_dim // 4, 1)
        
        # Additional components
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Process through LSTM layers
        current_input = x
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        
        for lstm_block in self.lstm_layers:
            current_input, h0, c0 = lstm_block(current_input, h0, c0)
        
        # Apply attention
        attended_output, _ = self.attention(current_input)
        
        # FC layers with residual connections
        x1 = self.fc1(attended_output)
        x1 = self.norm1(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        
        x2 = self.fc2(x1)
        x2 = self.norm2(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        
        x3 = self.fc3(x2)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)
        
        # Output
        out = self.sigmoid(self.output(x3))
        return out

class ModelTrainer:
    def __init__(self, input_dim, sequence_length=10, hidden_dim=128, num_layers=2):
        """
        Initialize the ModelTrainer.
        
        Args:
            input_dim (int): Number of input features
            sequence_length (int): Length of input sequences
            hidden_dim (int): Number of hidden units in LSTM
            num_layers (int): Number of LSTM layers
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.model = LSTMPredictor(input_dim, hidden_dim, num_layers).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.models_dir = Path(__file__).parent.parent.parent / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def prepare_sequences(self, data):
        """
        Prepare sequences for LSTM input.
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Sequences of data
        """
        sequences = []
        for i in range(len(data) - self.sequence_length):
            sequence = data[i:(i + self.sequence_length)]
            sequences.append(sequence)
        return np.array(sequences)

    def train(self, train_loader, val_loader, epochs=100):
        """
        Train the model.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Number of training epochs
            
        Returns:
            tuple: Lists of training and validation metrics
        """
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_path = self.models_dir / 'best_model.pth'
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        # Add L2 regularization
        weight_decay = 0.01
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=weight_decay)
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
                self.optimizer.step()
                
                total_loss += loss.item()
                predicted = (outputs.data > 0.5).float()
                total += batch_targets.size(0)
                correct += (predicted == batch_targets).sum().item()
            
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = 100 * correct / total
            
            # Validation
            self.model.eval()
            total_val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs = self.model(batch_features)
                    loss = self.criterion(outputs, batch_targets)
                    
                    total_val_loss += loss.item()
                    predicted = (outputs.data > 0.5).float()
                    total += batch_targets.size(0)
                    correct += (predicted == batch_targets).sum().item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            
            # Save metrics
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            
            logger.info(f'Epoch {epoch + 1}/{epochs}:')
            logger.info(f'Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
            logger.info(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f'Model saved to {best_model_path}')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(self, features):
        """
        Make predictions using the trained model.
        
        Args:
            features (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        self.model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(features).to(self.device)
            outputs = self.model(features)
            return outputs.cpu().numpy()

    def save_model(self, filename):
        """
        Save the model to disk.
        
        Args:
            filename (str): Name of the file to save the model
        """
        path = self.models_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f'Model saved to {path}')

    def load_model(self, filename):
        """
        Load a model from disk.
        
        Args:
            filename (str): Name of the file to load the model from
        """
        path = self.models_dir / filename
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f'Model loaded from {path}')

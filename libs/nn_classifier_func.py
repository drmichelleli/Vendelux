import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Define a neural network with dropout
class ImprovedNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out

class NNClassifier:
    def __init__(self, data_df):
        self.data_df = data_df
        self.random_seed = 42
        self.st_model = "paraphrase-MiniLM-L6-v2"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def run(self):
        start = datetime.now()
        # prepare the label
        
        self.data_df['label'] = self.data_df['IS_ATTENDING'].apply(lambda x: 1 if x == 'yes' else 0)
        self.data_df.reset_index(drop=True, inplace=True)

        # Define X and y
        X = self.data_df['text']
        y = self.data_df['label']

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

        # Encode the X_train and X_test with st_model
        model_txt = SentenceTransformer(self.st_model, device=self.device)
        X_train_encoded = model_txt.encode(X_train.tolist(), convert_to_tensor=True)
        X_test_encoded = model_txt.encode(X_test.tolist(), convert_to_tensor=True)

        # Normalize the encoded data
        X_train_encoded = (X_train_encoded - X_train_encoded.mean(dim=0)) / X_train_encoded.std(dim=0)
        X_test_encoded = (X_test_encoded - X_test_encoded.mean(dim=0)) / X_test_encoded.std(dim=0)

        # Convert y_train and y_test to tensors
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

        # Hyperparameters
        input_size = X_train_encoded.size(1)
        hidden_size1 = 256
        hidden_size2 = 128
        output_size = 2
        num_epochs = 30
        learning_rate = 0.001
        patience = 5  # Number of epochs to wait for improvement before stopping

        # Initialize the model, loss function, and optimizer
        model = ImprovedNN(input_size, hidden_size1, hidden_size2, output_size).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Early stopping variables
        best_loss = float('inf')
        epochs_no_improve = 0

        # Training loop with early stopping
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_encoded.to(self.device))
            loss = criterion(outputs, y_train_tensor.to(self.device))
            loss.backward()
            optimizer.step()
            
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_encoded.to(self.device))
                val_loss = criterion(val_outputs, y_test_tensor.to(self.device))
            
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            
            # Check for early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info('Early stopping!')
                    break

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_encoded.to(self.device))
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test_tensor.to(self.device)).sum().item() / y_test_tensor.size(0)
            logging.info(f'Accuracy: {accuracy:.4f}')

            # Print classification report
            class_report = classification_report(y_test_tensor.cpu(), predicted.cpu())
            logging.info(f'Classification Report:\n{class_report}')

        output_df = self.data_df.iloc[X_test.index]
        output_df['predicted'] = predicted.cpu().numpy()
        output_df['predicted'] = output_df['predicted'].apply(lambda x: 'yes' if x == 1 else 'no')

        logging.info(f'Training set size: {len(X_train)}')
        logging.info(f'Test set size: {len(X_test)}')

        end = datetime.now()
        time_taken = pd.Timedelta(end-start, unit='m') / timedelta(minutes=1)
        logging.info(f'Finished successfully and it took {time_taken} minutes')
        return output_df
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from libs.preprocess_func import merge_text
import logging
from datetime import datetime, timedelta

# define logger,file handler and set formatter
# logger = logging.getLogger(__name__)
# log_file = datetime.now().strftime('%Y_%m_%d_%H_%M')+'_nn_classifier.log'
# filename='./Logs/'+ log_file
# file_handler = logging.FileHandler(filename)
# logging.basicConfig(
#     format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
#     level=logging.INFO,
#     filename=filename,
# )

logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

start = datetime.now()
random_seed = 42
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# load the pre-trained SBERT model
st_model =  "paraphrase-MiniLM-L6-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_txt = SentenceTransformer(st_model, device=device)

# split the data to train and test
input_df = pd.read_excel('./input/Dataset.xlsx', sheet_name='post_details')
input_df.drop_duplicates(subset='POST_ID', inplace=True)
logging.info("input data size: %s",len(input_df))

# merge the Post text and the shared post text
input_df['post_date'] = input_df['DATE_PUBLISHED'].str[0:10]
input_df['post_date'] = input_df['post_date'].astype(str)
input_df['text'] = input_df.apply(merge_text, axis=1)
input_df['label'] = input_df['IS_ATTENDING'].apply(lambda x: 1 if x == 'yes' else 0)

# Define X and y
X = input_df['text']
y = input_df['label']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Encode the X_train and X_test with st_model
X_train_encoded = model_txt.encode(X_train.tolist(), convert_to_tensor=True)
X_test_encoded = model_txt.encode(X_test.tolist(), convert_to_tensor=True)

# Normalize the encoded data
X_train_encoded = (X_train_encoded - X_train_encoded.mean(dim=0)) / X_train_encoded.std(dim=0)
X_test_encoded = (X_test_encoded - X_test_encoded.mean(dim=0)) / X_test_encoded.std(dim=0)

# Convert y_train and y_test to tensors
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Define an improved neural network with dropout
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

# Hyperparameters
input_size = X_train_encoded.size(1)
hidden_size1 = 256
hidden_size2 = 128
output_size = 2
num_epochs = 30
learning_rate = 0.001
patience = 5  # Number of epochs to wait for improvement before stopping

# Initialize the model, loss function, and optimizer
model = ImprovedNN(input_size, hidden_size1, hidden_size2, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Early stopping variables
best_loss = float('inf')
epochs_no_improve = 0

# Training loop with early stopping
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_encoded.to(device))
    loss = criterion(outputs, y_train_tensor.to(device))
    loss.backward()
    optimizer.step()
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_encoded.to(device))
        val_loss = criterion(val_outputs, y_test_tensor.to(device))
    
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
    outputs = model(X_test_encoded.to(device))
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test_tensor.to(device)).sum().item() / y_test_tensor.size(0)
    logging.info(f'Accuracy: {accuracy:.4f}')

    # Print classification report
    class_report = classification_report(y_test_tensor.cpu(), predicted.cpu())
    logging.info(f'Classification Report:\n{class_report}')

logging.info(f'Training set size: {len(X_train)}')
logging.info(f'Test set size: {len(X_test)}')

end = datetime.now()
time_taken = pd.Timedelta(end-start, unit='m') / timedelta(minutes=1)
logging.info(f'Finished successfully and it took {time_taken} minutes')
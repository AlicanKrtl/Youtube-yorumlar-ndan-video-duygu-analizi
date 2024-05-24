import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Define the feedforward neural network model
class FFNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Define training function
def train(model, criterion, optimizer, train_loader, valid_loader, num_epochs=10, device='cpu'):
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        valid_loss, val_accuracy, report = evaluate(model, criterion, valid_loader, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {valid_loss:.4f}, Validation Acc: {val_accuracy:.4f}')
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saving best model...")
            
# Define evaluation function
def evaluate(model, criterion, data_loader, device='cpu'):
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = correct / total
    avg_loss = total_loss / len(data_loader)
    report = classification_report(all_labels, all_predictions)
    return avg_loss, accuracy, report

# Step 1: Load and preprocess data
data = pd.read_csv("/home/alican/Documents/Studies/begüm_proje/merged_data.csv")

# Step 2: Encode labels
label_encoder = LabelEncoder()
data['emotion_encoded'] = label_encoder.fit_transform(data['emotion'])

comments = data["comments"].apply(lambda x: " ".join(eval(x)) if len(x)>0 else "").values
emotion = data["emotion_encoded"].values

# Step 3: Vectorize text data
vectorizer = CountVectorizer(max_features=1000)  # Assuming you want to use a maximum of 1000 features
X = vectorizer.fit_transform(comments).toarray()

# Step 4: Split the dataset into train and test sets
X_train, X_valid, y_train, y_valid = train_test_split(X, emotion, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)

# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Define model parameters
input_dim = X_train.shape[1]  # Number of features
hidden_dim = 128
output_dim = len(np.unique(y_train))

# Instantiate the model
model = FFNClassifier(input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(model, criterion, optimizer, train_loader, valid_loader, num_epochs=10, device='cpu')

# Evaluate the best model
best_model = FFNClassifier(input_dim, hidden_dim, output_dim)
best_model.load_state_dict(torch.load('best_model.pth'))
best_model.eval()

# Evaluate the best model on val data
X_val_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_val_tensor = torch.tensor(y_valid, dtype=torch.long)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

val_loss, val_accuracy, val_report = evaluate(best_model, criterion, val_loader, device='cpu')

# Save results to a text file
with open('results.txt', 'w') as f:
    f.write(f'Validation Loss: {val_loss:.4f}\n')
    f.write(f'Validation Accuracy: {val_accuracy:.4f}\n')
    f.write(f'\nClassification Report:\n{val_report}\n')

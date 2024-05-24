import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load and preprocess data
data = pd.read_csv("/home/alican/Documents/Studies/begüm_proje/merged_data.csv")

# Step 2: Encode labels
label_encoder = LabelEncoder()
data['emotion_encoded'] = label_encoder.fit_transform(data['emotion'])

# Step 3: Train-test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Step 4: Vectorize text data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data['comments'])
X_test = vectorizer.transform(test_data['comments'])

# Convert to PyTorch tensors
X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_train = torch.tensor(train_data['emotion_encoded'].values, dtype=torch.long)
y_test = torch.tensor(test_data['emotion_encoded'].values, dtype=torch.long)

# Step 5: Define logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

# Step 6: Instantiate the model
input_dim = X_train.shape[1]
output_dim = len(label_encoder.classes_)
model = LogisticRegression(input_dim, output_dim)

# Step 7: Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Step 8: Train the model
num_epochs = 10
batch_size = 64
best_accuracy = 0

for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i+batch_size]
        labels = y_train[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Step 9: Evaluate the model
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy}")

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")
            with open("results.txt", "w") as f:
                f.write(f"Best Test Accuracy: {accuracy}\n")

# Step 10: Evaluate the best model
def evaluate_model(model, X_test, y_test, label_encoder):
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
        report = classification_report(y_test.numpy(), predicted.numpy(), target_names=label_encoder.classes_)
        return accuracy, report

best_model = LogisticRegression(input_dim, output_dim)
best_model.load_state_dict(torch.load("best_model.pth"))
accuracy, report = evaluate_model(best_model, X_test, y_test, label_encoder)

# Step 11: Save evaluation results to text file
with open("results.txt", "a") as f:
    f.write(f"Final Test Accuracy: {accuracy}\n")
    f.write("Classification Report:\n")
    f.write(report)

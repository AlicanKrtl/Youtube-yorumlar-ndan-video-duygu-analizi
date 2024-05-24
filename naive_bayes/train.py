import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Step 1: Load and preprocess data
data = pd.read_csv("/home/alican/Documents/Studies/begüm_proje/merged_data.csv")

# Step 2: Encode labels
label_encoder = LabelEncoder()
data['emotion_encoded'] = label_encoder.fit_transform(data['emotion'])

comments = data["comments"].apply(lambda x: " ".join(eval(x)) if len(x)>0 else "").values
emotion = data["emotion_encoded"].values

# Step 1: Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(comments, emotion, test_size=0.2, random_state=42)

# Step 2: Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 3: Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train_vec.toarray()).float()
X_test_tensor = torch.from_numpy(X_test_vec.toarray()).float()
y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)

# Step 4: Define Naive Bayes model
class NaiveBayes(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NaiveBayes, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.log_probs = nn.Parameter(torch.zeros(output_dim))
        self.log_likelihoods = nn.Parameter(torch.zeros((output_dim, input_dim)))

    def forward(self, x):
        log_prior = self.log_probs.unsqueeze(0)
        log_likelihood = torch.mm(x, self.log_likelihoods.t())
        log_posterior = log_prior + log_likelihood
        return log_posterior

# Step 5: Train the model
def train_model(model, criterion, optimizer, X_train, y_train, num_epochs=10):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

input_dim = X_train_tensor.shape[1]
output_dim = len(np.unique(y_train))
model = NaiveBayes(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor)

# Step 6: Evaluate the model
def evaluate_model(model, X_test, y_test, label_encoder):
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        predicted_classes = label_encoder.inverse_transform(predicted.numpy())
        y_test_classes = label_encoder.inverse_transform(y_test)
        return classification_report(y_test_classes, predicted_classes)

evaluation_results = evaluate_model(model, X_test_tensor, y_test_tensor, label_encoder)
print(evaluation_results)

# Step 7: Save the model and evaluation results to a text file
with open('naive_bayes_results.txt', 'w') as f:
    f.write("Evaluation Results:\n")
    f.write(evaluation_results)
    f.write("\n\nModel Parameters:\n")
    for param_tensor in model.state_dict():
        f.write(f"{param_tensor}\n")
        f.write(str(model.state_dict()[param_tensor]))
        f.write("\n")

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from utils import EmotionDataset, train_model, evaluate_model
from tqdm import tqdm

# Load data
csv_file = '/home/alican/Documents/Studies/begüm_proje/merged_data.csv'
dataset = EmotionDataset(csv_file)

# Split data into train and validation sets
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Create data loaders
batch_size = 16

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Load pre-trained BERT model and set up for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(dataset.labels))

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8, no_deprecation_warning=True)
total_steps = len(train_loader) * 3  # 3 epochs

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Fine-tune the model
train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=10)

# Evaluate the model on the validation set
accuracy, report = evaluate_model(model, val_loader, dataset)

# Save evaluation results to a text file
with open('evaluation_results.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy}\n')
    f.write('Classification Report:\n')
    f.write(report)
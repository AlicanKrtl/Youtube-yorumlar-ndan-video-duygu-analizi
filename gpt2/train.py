import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model, GPT2PreTrainedModel, GPT2Config
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Define a custom dataset class
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Define the GPT2ForSequenceClassification model
class GPT2ForSequenceClassification(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transformer = GPT2Model(config)
        self.classifier = nn.Linear(config.n_embd, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(last_hidden_states)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        return logits, loss

# Load and preprocess the data
data = pd.read_csv("/home/alican/Documents/Studies/begüm_proje/merged_data.csv")  # Replace "data.csv" with your CSV file path
texts = data['comments'].tolist()
labels = data['emotion'].astype('category').cat.codes.tolist()

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Initialize the GPT-2 tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
config = GPT2Config.from_pretrained('gpt2', num_labels=len(set(labels)))
model = GPT2ForSequenceClassification.from_pretrained('gpt2', config=config)
model.to(device)

# Create datasets and dataloaders
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length=128)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length=128)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Define training parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 10

best_val_accuracy = 0.0

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_dataloader)

    # Validation loop
    model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
    val_accuracy = accuracy_score(val_true, val_preds)
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_model.pth")

    print(f'Epoch {epoch + 1}:')
    print(f'Training Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Evaluate the best model
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
val_preds = []
val_true = []
with torch.no_grad():
    for batch in tqdm(val_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(logits, dim=1)
        val_preds.extend(preds.cpu().numpy())
        val_true.extend(labels.cpu().numpy())

# Get class names
class_names = data['emotion'].astype('category').cat.categories

# Save evaluation results with class names
report = classification_report(val_true, val_preds, target_names=class_names, zero_division=1)  # Setting zero_division parameter to 1
with open("evaluation_results.txt", "w") as f:
    f.write(report)

print("Evaluation results saved in evaluation_results.txt")

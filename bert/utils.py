import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, classification_report

class EmotionDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name='bert-base-uncased', max_length=128):
        self.data = pd.read_csv(csv_file)
        self.data["comments"] = self.data["comments"].apply(lambda x: " ".join(eval(x)) if len(x)>0 else "")
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.labels = {label: idx for idx, label in enumerate(self.data['emotion'].unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        text = self.data.iloc[idx, 1]
        emotion = self.data.iloc[idx, 2]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        label = self.labels[emotion]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=3, model_save_path='best_model.pth', results_save_path='training_results.txt'):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    best_val_accuracy = 0.0
    results = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_attention_mask = batch['attention_mask'].to(device)
            b_labels = batch['label'].to(device)

            model.zero_grad()

            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss}')

        # Validation
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                b_input_ids = batch['input_ids'].to(device)
                b_attention_mask = batch['attention_mask'].to(device)
                b_labels = batch['label'].to(device)

                outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
                loss = outputs.loss
                total_eval_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).flatten()
                accuracy = (preds == b_labels).cpu().numpy().mean() * 100
                total_eval_accuracy += accuracy

        avg_val_accuracy = total_eval_accuracy / len(val_loader)
        avg_val_loss = total_eval_loss / len(val_loader)

        print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}')
        
        # Save the best model
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            torch.save(model.state_dict(), model_save_path)

        # Save results
        results.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': avg_val_accuracy
        })

    # Save results to a text file
    with open(results_save_path, 'w') as f:
        for result in results:
            f.write(f"Epoch: {result['epoch']}, Train Loss: {result['train_loss']}, Validation Loss: {result['val_loss']}, Validation Accuracy: {result['val_accuracy']}\n")


# Evaluate the model
def evaluate_model(model, val_loader, dataset):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_attention_mask = batch['attention_mask'].to(device)
            b_labels = batch['label'].to(device)

            outputs = model(b_input_ids, attention_mask=b_attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).flatten()

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(b_labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=list(dataset.labels.keys()))

    print(f'Accuracy: {accuracy}')
    print(f'Classification Report:\n{report}')

if __name__ == "__main__":
    # Sample CSV file path
    file_path = '/home/alican/Documents/Studies/begüm_proje/merged_data.csv'


    dataset = EmotionDataset(file_path)
    print(dataset[0])
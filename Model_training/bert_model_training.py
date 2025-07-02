import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, classification_report
import pickle
from torch.optim import AdamW
import torch.nn as nn
from collections import Counter

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ChatbotDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors=None
        )
        
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_intents_data(json_file_path):
    """Load and prepare data from intents JSON file"""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        intents = json.load(file)
    
    texts = []
    labels = []
    
    for intent in intents['intents']:
        tag = intent['tag']
        patterns = intent['patterns']
        
        for pattern in patterns:
            texts.append(pattern)
            labels.append(tag)
    
    return texts, labels

def train_bert_model(model, train_loader, val_loader, device, num_epochs=15):
    """Train the BERT model"""
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8, weight_decay=0.01)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps * 0.1,
        num_training_steps=total_steps
    )
    
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Average training loss: {avg_train_loss:.4f}')
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_val_loss += outputs.loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = accuracy_score(true_labels, predictions)
        
        print(f'Average validation loss: {avg_val_loss:.4f}')
        print(f'Validation accuracy: {val_accuracy:.4f}')
        print('-' * 50)
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'model/bert_chatbot_model.pth')
            print(f'New best model saved with accuracy: {best_accuracy:.4f}')

    # Freeze BERT layers, only train classifier
    for param in model.bert.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

class EnhancedDistilBERTClassifier(nn.Module):
    def __init__(self, distilbert_model, num_labels, dropout_rate=0.3):
        super().__init__()
        self.distilbert = distilbert_model
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(distilbert_model.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        distilbert_outputs = self.distilbert.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = distilbert_outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
        return type('Outputs', (), {'loss': loss, 'logits': logits})()

def train_distilbert_model_enhanced(model, train_loader, val_loader, device, label_encoder, num_epochs=20, patience=5):
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    best_accuracy = 0
    best_epoch = 0
    patience_counter = 0
    training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    print(f"Training for {num_epochs} epochs with patience={patience}")
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        model.train()
        total_train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        avg_train_loss = total_train_loss / len(train_loader)
        training_history['train_loss'].append(avg_train_loss)
        print(f'Average training loss: {avg_train_loss:.4f}')
        model.eval()
        total_val_loss = 0
        predictions = []
        true_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_val_loss += outputs.loss.item()
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = accuracy_score(true_labels, predictions)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_accuracy'].append(val_accuracy)
        print(f'Average validation loss: {avg_val_loss:.4f}')
        print(f'Validation accuracy: {val_accuracy:.4f}')
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), 'model/Distilled_model.pth')
            print(f'✅ New best model weights saved with accuracy: {best_accuracy:.4f}')
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epochs')
        if (epoch + 1) % 5 == 0:
            print("\nClassification Report:")
            print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))
        print('-' * 60)
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            print(f'Best accuracy: {best_accuracy:.4f} at epoch {best_epoch}')
            break
    return training_history, best_accuracy, best_epoch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    json_file_path = 'Model_training/chatbot_intents.json'
    texts, labels = load_intents_data(json_file_path)
    print(f'Total samples: {len(texts)}')
    print(f'Unique labels: {len(set(labels))}')
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    X_train, X_val, y_train, y_val = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    print(f'Training samples: {len(X_train)}')
    print(f'Validation samples: {len(X_val)}')
    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name, use_auth_token=False)
    base_model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_),
        output_attentions=False,
        output_hidden_states=False,
        use_auth_token=False
    )
    model = EnhancedDistilBERTClassifier(base_model, len(label_encoder.classes_), dropout_rate=0.3)
    model.to(device)
    train_dataset = ChatbotDataset(X_train, y_train, tokenizer)
    val_dataset = ChatbotDataset(X_val, y_val, tokenizer)
    batch_size = 12 # 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    os.makedirs('model', exist_ok=True)
    print('Starting DistilBERT fine-tuning...')
    training_history, best_accuracy, best_epoch = train_distilbert_model_enhanced(model, train_loader, val_loader, device, label_encoder, num_epochs=20, patience=5)
    tokenizer.save_pretrained('model/')
    with open('model/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open('model/label_mapping.pkl', 'wb') as f:
        pickle.dump(label_mapping, f)
    print('Training completed!')
    print(f'Best accuracy: {best_accuracy:.4f} at epoch {best_epoch}')
    print('Saved files:')
    print('- model/Distilled_model.pth (best model weights)')
    print('- model/label_encoder.pkl (label encoder)')
    print('- model/label_mapping.pkl (label mapping)')
    print('- model/ (tokenizer files)')

if __name__ == '__main__':
    main() 
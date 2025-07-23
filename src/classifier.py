import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import mlflow
import json
import os

# Force CUDA usage for RTX 3090 Ti
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(0)  # Use first GPU
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    device = torch.device('cpu')
    print('CUDA not available, falling back to CPU')

# Define the Dataset class for loading
class Dataset(Dataset):
    # This class handles the dataset for bug descriptions and their severity labels.
    def __init__(self, texts, labels, tokenizer, max_length=512):
        # Initialize the dataset with texts, labels, tokenizer, and max length.
        self.texts = texts
        # Store the labels, tokenizer, and maximum length for tokenization.
        self.labels = labels
        # Initialize the tokenizer and set the maximum length for tokenization.
        self.tokenizer = tokenizer
        # Set the maximum length for tokenization.
        self.max_length = max_length

    # Define the length of the dataset.    
    def __len__(self):
        return len(self.texts)
    
    # This method retrieves the item at the specified index.
    def __getitem__(self, idx):
        # Tokenize the text at the specified index.
        encoding = self.tokenizer(
            str(self.texts[idx]), # Convert the text to string to ensure compatibility.
            truncation=True, # Truncate the text if it exceeds the maximum length.
            padding='max_length', # Pad the text to the maximum length.
            max_length=self.max_length, # Set the maximum length for tokenization.
            return_tensors='pt' # Return the tokenized text as PyTorch tensors.
        )
        
        # Return a dictionary containing the input IDs, attention mask, and label.
        return {
            'input_ids': encoding['input_ids'].flatten(), # Flatten the input IDs tensor to a 1D tensor.
            'attention_mask': encoding['attention_mask'].flatten(), # Flatten the attention mask tensor to a 1D tensor.
            'label': torch.tensor(self.labels[idx], dtype=torch.long) # Convert the label to a PyTorch tensor of type long.
        }

# Define the Classifier class for training and evaluation
class Classifier:
    # This class handles the training and evaluation of a BERT model for bug severity classification.
    def __init__(self):
        # Initialize the tokenizer and model from the pre-trained BERT base uncased model.
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load the BERT model for sequence classification with 5 labels (severity levels).
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', # Load the pre-trained BERT model.
            num_labels=5 # Specify the number of labels for classification (5 severity levels: 1-5).
        )
        
        # Move the model to GPU immediately
        self.model.to(device)
        
        # Enable mixed precision for RTX 3090 Ti
        if device.type == 'cuda':
            print(f'Model loaded on GPU: {torch.cuda.get_device_name(0)}')
            print(f'GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB')
            # Clear any cached memory
            torch.cuda.empty_cache()
        else:
            print(f'Model loaded on: {device}')

    # This method prepares the data for training and validation.        
    def prepare_data(self, df):
        # Convert 1-5 to 0-4 for model so the data is normalized for classification.
        df['label'] = df['severity'] - 1
        
        # Split the dataset into training and validation sets (80% train, 20% validation).
        X_train, X_val, y_train, y_val = train_test_split(
            df['description'].values, # Use the 'description' column for training.
            df['label'].values, # Use the 'label' column for training.
            test_size=0.2, # Set the test size to 20% for validation.
            random_state=42, # Set a random seed for reproducibility.
            stratify=df['label'].values # Stratify the split based on labels to maintain class distribution.
        )
        
        # Create the training dataset using the Dataset class.
        train_dataset = Dataset(X_train, y_train, self.tokenizer)
        
        # Create the validation dataset using the Dataset class.
        val_dataset = Dataset(X_val, y_val, self.tokenizer)
        
        # Return the training and validation datasets, and the validation features and labels for later evaluation.
        return train_dataset, val_dataset, X_val, y_val
    
    # This method trains the model on the training dataset and evaluates it on the validation dataset.
    def train(self, train_dataset, val_dataset, epochs=3, batch_size=None):
        # Optimize batch size for RTX 3090 Ti (24GB VRAM)
        if batch_size is None:
            if device.type == 'cuda':
                # RTX 3090 Ti can handle larger batch sizes
                batch_size = 32  # Increased from 16 for better GPU utilization
                print(f"Using optimized batch size for RTX 3090 Ti: {batch_size}")
            else:
                batch_size = 16
        
        # Start MLflow run
        mlflow.start_run()
        
        # Log parameters for MLflow
        mlflow.log_param("model_type", "bert-base-uncased")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", 2e-5)
        mlflow.log_param("device", str(device))
        
        # Create data loaders with optimized settings for GPU
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4 if device.type == 'cuda' else 0,  # Use multiple workers for GPU
            pin_memory=True if device.type == 'cuda' else False  # Pin memory for faster GPU transfer
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            num_workers=4 if device.type == 'cuda' else 0,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # Set the model to training mode
        optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        
        # Enable mixed precision training for RTX 3090 Ti
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        # Initialize history dictionary to store training and validation metrics
        history = {'train_loss': [], 'val_accuracy': []}
        
        # Loop through the specified number of epochs
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            
            # Set the model to training mode
            self.model.train()
            
            # Initialize the training loss for this epoch
            train_loss = 0
            
            # Clear GPU cache at start of each epoch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                print(f'GPU Memory before epoch: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB')
            
            # Iterate over the training data loader
            for batch_idx, batch in enumerate(tqdm(train_loader, desc='Training')):
                optimizer.zero_grad()
                
                # Move batch to device with non_blocking for faster transfer
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                
                # Use mixed precision for RTX 3090 Ti
                if device.type == 'cuda' and scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                    
                    # Scale the loss and backward pass
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard training for CPU
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                train_loss += loss.item()
                
                # Print GPU memory usage every 100 batches
                if device.type == 'cuda' and batch_idx % 100 == 0:
                    print(f'Batch {batch_idx}: GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB')
            
            # Calculate average training loss for the epoch
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            val_accuracy = self.evaluate(val_loader)
            history['val_accuracy'].append(val_accuracy)
            
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Val Accuracy: {val_accuracy:.4f}')
            
            if device.type == 'cuda':
                print(f'Max GPU Memory Used: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB')
                torch.cuda.reset_peak_memory_stats()  # Reset for next epoch
            
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
        
        mlflow.end_run()
        return history
    
    def evaluate(self, data_loader):
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                # Move batch to device with non_blocking
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                
                # Use mixed precision for inference if available
                if device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                _, preds = torch.max(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch['label'].numpy())
        
        return accuracy_score(true_labels, predictions)
    
    def save_model(self, path='model.pt'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer
        }, path)
        print(f'Model saved to {path}')
    
    def generate_classification_report(self, X_val, y_val, save_path='metrics/'):
        os.makedirs(save_path, exist_ok=True)
        
        # Get predictions with GPU optimization
        self.model.eval()
        predictions = []
        
        # Process in batches for better GPU utilization
        batch_size = 64 if device.type == 'cuda' else 8
        
        for i in tqdm(range(0, len(X_val), batch_size), desc='Generating predictions'):
            batch_texts = X_val[i:i+batch_size]
            
            # Tokenize batch
            batch_encodings = self.tokenizer(
                list(batch_texts),
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                input_ids = batch_encodings['input_ids'].to(device, non_blocking=True)
                attention_mask = batch_encodings['attention_mask'].to(device, non_blocking=True)
                
                if device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                _, batch_preds = torch.max(outputs.logits, dim=1)
                predictions.extend(batch_preds.cpu().numpy())
        
        # Classification report
        report = classification_report(
            y_val, predictions,
            target_names=['SEV-1 Critical', 'SEV-2 Major', 'SEV-3 Minor', 'SEV-4 Low', 'SEV-5 Trivial'],
            output_dict=True
        )
        
        # Save report
        with open(f'{save_path}classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Confusion matrix
        cm = confusion_matrix(y_val, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['SEV-1 Critical', 'SEV-2 Major', 'SEV-3 Minor', 'SEV-4 Low', 'SEV-5 Trivial'],
                    yticklabels=['SEV-1 Critical', 'SEV-2 Major', 'SEV-3 Minor', 'SEV-4 Low', 'SEV-5 Trivial'])
        plt.title('Confusion Matrix - 5-Tier Severity Classification')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{save_path}confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nClassification Report:")
        print(classification_report(
            y_val, predictions,
            target_names=['SEV-1 Critical', 'SEV-2 Major', 'SEV-3 Minor', 'SEV-4 Low', 'SEV-5 Trivial']
        ))
        
        return report
    
# Add this at the very bottom of bug_severity_classifier.py
if __name__ == "__main__":
    # Print system information
    print("="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    print("="*60)
    
    # Load data
    print("Loading training data...")
    df = pd.read_excel('dataset.xlsx')
    print(f"Loaded {len(df)} bug descriptions")
    print(f"Severity distribution:\n{df['severity'].value_counts().sort_index()}")
    
    # Initialize classifier
    classifier = Classifier()
    
    # Prepare data
    train_dataset, val_dataset, X_val, y_val = classifier.prepare_data(df)
    
    # Train model with GPU optimization
    print("\nStarting training...")
    if device.type == 'cuda':
        print("Training on RTX 3090 Ti - Expected time: 3-5 minutes")
    else:
        print("Training on CPU - Expected time: 8-15 minutes")
    
    history = classifier.train(train_dataset, val_dataset, epochs=3)
    
    # Save model
    classifier.save_model('model.pt')
    
    # Generate classification report and confusion matrix
    print("\nGenerating performance metrics...")
    classifier.generate_classification_report(X_val, y_val)
    
    print("\n" + "="*60)
    print("Training complete! Model saved to model.pt")
    print("Check the 'metrics' folder for the confusion matrix and classification report")
    if device.type == 'cuda':
        print(f"Final GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print("="*60)
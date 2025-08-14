#!/usr/bin/env python3
# Suppress TensorFlow logging
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Libraries
import torch
from torch.optim import AdamW
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import shutil
import umap.umap_ as umap

# Verify GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def setup_logging():
    """Configure logging to file and console."""
    log_file = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_model(model, tokenizer, output_dir, iteration="final"):
    """Save model and tokenizer to disk with metadata."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    with open(os.path.join(output_dir, "training_info.txt"), "w") as f:
        f.write(f"Iteration: {iteration}\n")
        f.write(f"Saved at: {datetime.now()}\n")
    
    logging.info(f"Model saved to {output_dir} (Iteration {iteration})")

def plot_embeddings(embeddings, labels, iteration, epoch):
    """Visualize embeddings using UMAP."""
    reducer = umap.UMAP(random_state=42)
    proj = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='Spectral', s=10)
    plt.colorbar(scatter, label='Class Labels')
    plt.title(f'Embedding Space (Iter {iteration}, Epoch {epoch})')
    
    os.makedirs("embeddings_plots", exist_ok=True)
    plt.savefig(f"embeddings_plots/embedding_iter{iteration}_epoch{epoch}.png")
    plt.close()

def topic_informed_loss(logits, labels, topic_probs, class_weights, device, alpha=0.7):
    """Compute topic-informed loss combining cross-entropy and KL divergence."""
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))(logits, labels)
    soft_loss = torch.nn.KLDivLoss(reduction='batchmean')(torch.nn.functional.log_softmax(logits, dim=-1), topic_probs)
    return alpha * ce_loss + (1 - alpha) * soft_loss

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, topic_probs, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.topic_probs = topic_probs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        topic_probs = self.topic_probs[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'topic_probs': torch.tensor(topic_probs, dtype=torch.float),
            'text': text
        }

def train_model(model, dataloader, optimizer, scheduler, device, epoch, epochs, class_weights):
    """Train with topic-informed loss and learning rate warmup."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device)
        }
        labels = batch['labels'].to(device)
        topic_probs = batch['topic_probs'].to(device)
        
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = topic_informed_loss(outputs.logits, labels, topic_probs, class_weights, device)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate_classification_metrics(model, dataloader, device, iteration, epoch, topic_to_label):
    """Calculate accuracy, precision, recall, per-class F1, and visualize embeddings."""
    model.eval()
    all_preds = []
    all_labels = []
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            labels = batch['labels'].to(device)
            outputs = model(**inputs, output_hidden_states=True)
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            all_embeddings.extend(embeddings)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_embeddings = np.array(all_embeddings)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Per-class F1 scores
    per_class_f1 = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )[2]
    label_to_topic = {v: k for k, v in topic_to_label.items()}
    for i, f1_score in enumerate(per_class_f1):
        logging.info(f"F1 Score for {label_to_topic[i]}: {f1_score:.4f}")
    
    if len(all_embeddings) > 0:
        plot_embeddings(all_embeddings, all_labels, iteration, epoch)
    
    # Log only the overall metrics
    logging.info(
        f"Metrics - Accuracy: {accuracy:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1-Score: {f1:.4f}"
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class_f1': {label_to_topic[i]: f1_score for i, f1_score in enumerate(per_class_f1)}
    }

def check_data_leakage(train_df, val_df):
    """Verify no overlap between train/val splits."""
    train_texts = set(train_df['text'].astype(str))
    val_texts = set(val_df['text'].astype(str))
    overlap = train_texts & val_texts
    if overlap:
        logging.warning(f"Data leakage detected! {len(overlap)} overlapping samples: {list(overlap)[:5]}")
    else:
        logging.info("No data leakage detected")

def main():
    setup_logging()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    if device.type == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    # Argument parsing
    parser = argparse.ArgumentParser(description='BERT Training with Labeled Data')
    parser.add_argument('--labeled_data', type=str, default='../../datasets/labeled_reviews.csv',
                        help='Path to labeled dataset with text, label, and topic_probs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs per iteration')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps')
    parser.add_argument('--output_dir', type=str, default='saved_models',
                        help='Directory to save models')
    args = parser.parse_args()

    # Load labeled data
    logging.info("Loading labeled data...")
    try:
        labeled_df = pd.read_csv(args.labeled_data)
        if 'text' not in labeled_df.columns:
            labeled_df = labeled_df.rename(columns={'processed_reviews': 'text'})
        if 'topic_probs' not in labeled_df.columns:
            logging.error("Labeled data must contain 'topic_probs' column")
            return
        labeled_df['topic_probs'] = labeled_df['topic_probs'].apply(lambda x: np.array([float(p) for p in x.split(',')]))
        logging.info(f"Labeled data loaded: {len(labeled_df)} samples")
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return

    # Model initialization
    logging.info("Initializing model...")
    try:
        tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        model = BertForSequenceClassification.from_pretrained(
            'huawei-noah/TinyBERT_General_4L_312D',
            num_labels=3  # Adjust based on expected number of topics
        ).to(device)
    except Exception as e:
        logging.error(f"Model initialization failed: {str(e)}")
        return

    # Create topic_to_label mapping
    topic_to_label = {f"topic_{i}": i for i in range(3)}  # Adjust based on expected number of topics

    # Compute class weights
    N = len(labeled_df)
    C = len(topic_to_label)
    class_counts = labeled_df['label'].value_counts().to_dict()
    class_weights = np.zeros(C)
    for label in range(C):
        count = class_counts.get(label, 1)  # Avoid division by zero
        class_weights[label] = N / (count * C)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    logging.info("\n=== Class Distribution and Weights ===")
    for topic, label in topic_to_label.items():
        count = class_counts.get(label, 0)
        weight = class_weights[label].item()
        logging.info(f"Label {label}: '{topic}' - {count} samples ({count/N:.1%}), Weight: {weight:.4f}")

    # Split labeled data
    train_df, val_df = train_test_split(
        labeled_df,
        test_size=0.2,
        random_state=42,
        stratify=labeled_df['label']
    )
    check_data_leakage(train_df, val_df)
    
    # Create DataLoaders
    train_loader = DataLoader(
        ReviewDataset(train_df['text'].values, train_df['label'].values, train_df['topic_probs'].values, tokenizer),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        ReviewDataset(val_df['text'].values, val_df['label'].values, val_df['topic_probs'].values, tokenizer),
        batch_size=args.batch_size,
        pin_memory=True
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_loader) * args.epochs
    )

    # Training loop
    prev_f1 = 0.0
    f1_threshold = 0.01
    for iteration in range(3):
        logging.info(f"\n=== Iteration {iteration + 1} ===")
        
        for epoch in range(args.epochs):
            train_loss = train_model(model, train_loader, optimizer, scheduler, device, epoch, args.epochs, class_weights)
            logging.info(f"Epoch {epoch + 1} Loss: {train_loss:.4f}")
            
            metrics = evaluate_classification_metrics(model, val_loader, device, iteration + 1, epoch + 1, topic_to_label)
            
        # Save intermediate model
        save_model(model, tokenizer, 
                 os.path.join(args.output_dir, f"iter_{iteration + 1}"), 
                 iteration + 1)
        
        if abs(metrics['f1_score'] - prev_f1) < f1_threshold and metrics['f1_score'] >= 0.7:
            logging.info(f"Model stabilized (F1 change < {f1_threshold:.3f}, F1 = {metrics['f1_score']:.4f}). Stopping training.")
            break
        prev_f1 = metrics['f1_score']

    # Final evaluation and model save
    final_metrics = evaluate_classification_metrics(model, val_loader, device, iteration + 1, "final", topic_to_label)
    logging.info(
        f"\nFinal Metrics - Accuracy: {final_metrics['accuracy']:.4f} | "
        f"Precision: {final_metrics['precision']:.4f} | "
        f"Recall: {final_metrics['recall']:.4f} | "
        f"F1-Score: {final_metrics['f1_score']:.4f}"
    )
    
    save_model(model, tokenizer, os.path.join(args.output_dir, "final_model"))
    logging.info("Training completed successfully")

if __name__ == "__main__":
    main()
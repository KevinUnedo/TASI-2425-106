#!/usr/bin/env python3
import torch
from torch.optim import AdamW
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
import os
from collections import defaultdict

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

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
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
            'text': text  # Keep text for pseudo-labeling
        }

def train_model(model, dataloader, optimizer, scheduler, device, epoch, epochs):
    """Train with learning rate warmup."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
            'labels': batch['labels'].to(device)
        }
        
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate_cosine_similarity(model, dataloader, device):
    """Calculate intra-class cosine similarity."""
    model.eval()
    class_embeddings = defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            outputs = model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            
            for emb, label in zip(embeddings, batch['labels']):
                class_embeddings[label.item()].append(emb)
    
    # Calculate intra-class similarity
    intra_sim = []
    for label, embs in class_embeddings.items():
        if len(embs) > 1:  # Need at least 2 samples
            sim_matrix = cosine_similarity(embs)
            mask = np.triu_indices(len(embs), k=1)  # Upper triangle
            intra_sim.append(sim_matrix[mask].mean())
    
    return np.mean(intra_sim) if intra_sim else 0.0

def generate_pseudo_labels(model, unlabeled_loader, device, threshold=0.8):
    """Generate high-confidence pseudo-labels."""
    model.eval()
    pseudo_data = []
    
    with torch.no_grad():
        for batch in tqdm(unlabeled_loader, desc="Pseudo-labeling"):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)
            
            for text, pred, conf in zip(batch['text'], preds, confidences):
                if conf > threshold:
                    pseudo_data.append({
                        'text': text,
                        'label': pred.item(),
                        'confidence': conf.item()
                    })
    
    return pd.DataFrame(pseudo_data)

def check_data_leakage(train_df, val_df):
    """Verify no overlap between train/val splits."""
    train_texts = set(train_df['text'].astype(str))
    val_texts = set(val_df['text'].astype(str))
    overlap = train_texts & val_texts
    if overlap:
        logging.warning(f"Data leakage detected! {len(overlap)} overlapping samples")
    else:
        logging.info("No data leakage detected")

def main():
    setup_logging()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    if device.type == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    # Argument parsing (removed sample_frac)
    parser = argparse.ArgumentParser(description='BERT Training with Pseudo-Labeling')
    parser.add_argument('--pseudo_data', type=str, required=True,
                      help='Path to pseudo-labeled data')
    parser.add_argument('--unlabeled_data', type=str, required=True,
                      help='Path to unlabeled data')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs per iteration')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                      help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=100,
                      help='Number of warmup steps')
    parser.add_argument('--threshold', type=float, default=0.8,
                      help='Confidence threshold for pseudo-labeling')
    args = parser.parse_args()

    # Data loading - USING 100% OF PSEUDO DATA
    logging.info("Loading ALL pseudo-labeled data...")
    try:
        pseudo_df = pd.read_csv(args.pseudo_data)
        topic_to_label = {topic: idx for idx, topic in enumerate(pseudo_df['topic'].unique())}
        pseudo_df['label'] = pseudo_df['topic'].map(topic_to_label)
        
        # Split ALL data (no sampling)
        train_df, val_df = train_test_split(
            pseudo_df,  # Using full dataset
            test_size=0.2,
            random_state=42,
            stratify=pseudo_df['label']
        )
        check_data_leakage(train_df, val_df)
        
        # Load unlabeled data
        unlabeled_df = pd.read_csv(args.unlabeled_data)
        logging.info(f"Data loaded - Train: {len(train_df)}, Val: {len(val_df)}, Unlabeled: {len(unlabeled_df)}")
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return

    # Model initialization
    logging.info("Initializing model...")
    try:
        tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        model = BertForSequenceClassification.from_pretrained(
            'huawei-noah/TinyBERT_General_4L_312D',
            num_labels=len(topic_to_label)
        ).to(device)
    except Exception as e:
        logging.error(f"Model initialization failed: {str(e)}")
        return

    # DataLoaders
    train_loader = DataLoader(
        ReviewDataset(train_df['text'].values, train_df['label'].values, tokenizer),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        ReviewDataset(val_df['text'].values, val_df['label'].values, tokenizer),
        batch_size=args.batch_size,
        pin_memory=True
    )
    unlabeled_loader = DataLoader(
        ReviewDataset(unlabeled_df['processed_reviews'].values, 
                     np.zeros(len(unlabeled_df)), tokenizer),
        batch_size=args.batch_size * 2,  # Larger batches for pseudo-labeling
        pin_memory=True
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_loader) * args.epochs
    )

    # Training loop with pseudo-labeling
    for iteration in range(3):  # 3 pseudo-labeling iterations
        logging.info(f"\n=== Iteration {iteration + 1} ===")
        
        # Training
        for epoch in range(args.epochs):
            train_loss = train_model(model, train_loader, optimizer, scheduler,  device,
                                   epoch, args.epochs)
            logging.info(f"Epoch {epoch + 1} Loss: {train_loss:.4f}")
        
        # Evaluation
        cosine_sim = evaluate_cosine_similarity(model, val_loader, device)
        logging.info(f"Intra-class Cosine Similarity: {cosine_sim:.4f}")
        
        # Pseudo-labeling if similarity is low
        if cosine_sim < 0.5:  # Threshold for adding pseudo-labels (next update to 0.8)
            logging.info("Generating pseudo-labels...")
            pseudo_df = generate_pseudo_labels(model, unlabeled_loader, 
                                             device, args.threshold)
            
            if len(pseudo_df) > 0:
                # Add to training data
                train_df = pd.concat([
                    train_df,
                    pd.DataFrame({
                        'text': pseudo_df['text'],
                        'label': pseudo_df['label']
                    })
                ])
                
                # Update DataLoader
                train_loader = DataLoader(
                    ReviewDataset(train_df['text'].values, 
                                train_df['label'].values, tokenizer),
                    batch_size=args.batch_size,
                    shuffle=True,
                    pin_memory=True
                )
                logging.info(f"Added {len(pseudo_df)} pseudo-labels (Total train: {len(train_df)})")
            else:
                logging.warning("No high-confidence pseudo-labels found")

    # Final evaluation
    final_sim = evaluate_cosine_similarity(model, val_loader, device)
    logging.info(f"\nFinal Intra-class Cosine Similarity: {final_sim:.4f}")
    logging.info("Training completed successfully")

if __name__ == "__main__":
    main()
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
import matplotlib.pyplot as plt
import umap.umap_ as umap
import shutil  # For directory operations

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
    # Remove existing directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training metadata
    with open(os.path.join(output_dir, "training_info.txt"), "w") as f:
        f.write(f"Iteration: {iteration}\n")
        f.write(f"Saved at: {datetime.now()}\n")
    
    logging.info(f"Model saved to {output_dir} (Iteration {iteration})")

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
            'text': text
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

def evaluate_cosine_similarity(model, dataloader, device, iteration, epoch):
    """Calculate intra-class and inter-class cosine similarity."""
    model.eval()
    class_embeddings = defaultdict(list)
    all_embeddings = []
    all_labels = []
    
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
                all_embeddings.append(emb)
                all_labels.append(label.item())
    
    # Convert to numpy arrays
    class_embeddings = {k: np.vstack(v) for k, v in class_embeddings.items()}
    class_labels = list(class_embeddings.keys())
    
    # --- Intra-Class Similarity ---
    intra_sim = []
    for label, embs in class_embeddings.items():
        if len(embs) > 1:
            sim_matrix = cosine_similarity(embs)
            mask = np.triu_indices(len(embs), k=1)
            intra_sim.append(sim_matrix[mask].mean())
    intra_sim_mean = np.mean(intra_sim) if intra_sim else 0.0
    
    # --- Inter-Class Similarity ---
    inter_sim = []
    if len(class_labels) > 1:
        for i, label1 in enumerate(class_labels[:-1]):
            embs1 = class_embeddings[label1]
            for label2 in class_labels[i+1:]:
                embs2 = class_embeddings[label2]
                sim_matrix = cosine_similarity(embs1, embs2)
                inter_sim.append(sim_matrix.mean())
        inter_sim_mean = np.mean(inter_sim) if inter_sim else 0.0
    else:
        inter_sim_mean = 0.0
    
    # --- Visualization ---
    if len(all_embeddings) > 0:
        plot_embeddings(np.array(all_embeddings), np.array(all_labels), iteration, epoch)
    
    return {
        'intra_class_sim': intra_sim_mean,
        'inter_class_sim': inter_sim_mean,
        'separation_score': intra_sim_mean - inter_sim_mean
    }

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

    # Argument parsing
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
    parser.add_argument('--output_dir', type=str, default='saved_models',
                      help='Directory to save models')
    args = parser.parse_args()

    # Data loading
    logging.info("Loading ALL pseudo-labeled data...")
    try:
        pseudo_df = pd.read_csv(args.pseudo_data)
        topic_to_label = {topic: idx for idx, topic in enumerate(pseudo_df['topic'].unique())}
        pseudo_df['label'] = pseudo_df['topic'].map(topic_to_label)
        
        # Log class distribution
        logging.info("\n=== Class Distribution ===")
        for topic, label in topic_to_label.items():
            count = len(pseudo_df[pseudo_df['label'] == label])
            logging.info(f"Label {label}: '{topic}' - {count} samples ({count/len(pseudo_df):.1%})")
        
        # Split data
        train_df, val_df = train_test_split(
            pseudo_df,
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
        batch_size=args.batch_size * 2,
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
        
        for epoch in range(args.epochs):
            # Training
            train_loss = train_model(model, train_loader, optimizer, scheduler, device, epoch, args.epochs)
            logging.info(f"Epoch {epoch + 1} Loss: {train_loss:.4f}")
            
            # Evaluation (every epoch for visualization)
            metrics = evaluate_cosine_similarity(model, val_loader, device, iteration + 1, epoch + 1)
            logging.info(
                f"Metrics - Intra: {metrics['intra_class_sim']:.4f} | "
                f"Inter: {metrics['inter_class_sim']:.4f} | "
                f"Separation: {metrics['separation_score']:.4f}"
            )
        
        # Save intermediate model
        save_model(model, tokenizer, 
                 os.path.join(args.output_dir, f"iter_{iteration + 1}"), 
                 iteration + 1)
        
        # Pseudo-labeling condition
        if metrics['separation_score'] < 0.3:
            logging.info("Generating pseudo-labels...")
            pseudo_df = generate_pseudo_labels(model, unlabeled_loader, device, args.threshold)
            
            if len(pseudo_df) > 0:
                train_df = pd.concat([train_df, pseudo_df[['text', 'label']]])
                train_loader = DataLoader(
                    ReviewDataset(train_df['text'].values, train_df['label'].values, tokenizer),
                    batch_size=args.batch_size,
                    shuffle=True,
                    pin_memory=True
                )
                logging.info(f"Added {len(pseudo_df)} pseudo-labels (Total train: {len(train_df)})")
            else:
                logging.warning("No high-confidence pseudo-labels found")

    # Final evaluation and model save
    final_metrics = evaluate_cosine_similarity(model, val_loader, device, iteration + 1, "final")
    logging.info(
        f"\nFinal Metrics - Intra: {final_metrics['intra_class_sim']:.4f} | "
        f"Inter: {final_metrics['inter_class_sim']:.4f} | "
        f"Separation: {final_metrics['separation_score']:.4f}"
    )
    
    save_model(model, tokenizer, os.path.join(args.output_dir, "final_model"))
    logging.info("Training completed successfully")

if __name__ == "__main__":
    main()
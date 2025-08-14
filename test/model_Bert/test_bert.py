#!/usr/bin/env python3
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

def setup_logging():
    log_file = f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

class ReviewDataset(Dataset):
    def __init__(self, texts, topic_probs, tokenizer, max_length=128):
        self.texts = texts
        self.topic_probs = topic_probs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
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
            'topic_probs': torch.tensor(topic_probs, dtype=torch.float),
            'text': text
        }

def plot_embeddings(embeddings, labels, topic_names, plot_dir, iteration="final"):
    logging.info("Starting embedding visualization...")
    try:
        logging.info("Checking embeddings shape...")
        logging.info(f"Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}")
        if embeddings.shape[0] < 2:
            raise ValueError("Need at least 2 samples for UMAP visualization")
        
        logging.info("Normalizing embeddings for UMAP...")
        embeddings = normalize(embeddings, norm='l2')
        
        logging.info("Initializing UMAP reducer with adjusted parameters...")
        reducer = umap.UMAP(random_state=42, n_neighbors=30, min_dist=0.05, metric='cosine')
        logging.info("Performing UMAP reduction...")
        proj = reducer.fit_transform(embeddings)
        logging.info(f"UMAP reduction completed. Shape of projection: {proj.shape}")
        
        logging.info("Creating scatter plot...")
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='Spectral', s=10, alpha=0.6)
        plt.colorbar(scatter, label='Topic Labels')
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=sns.color_palette('Spectral', len(topic_names))[i], 
                              markersize=10, label=topic_names[i]) for i, name in enumerate(topic_names)]
        plt.legend(handles=handles, title='Topics')
        plt.title(f'TinyBERT Embedding Space (Iteration {iteration})')
        
        plot_path = os.path.join(plot_dir, f'tinybert_scatter_iter_{iteration}.png')
        logging.info(f"Attempting to save plot to {plot_path}")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(f"TinyBERT scatter plot saved to {plot_path}")
        print(f"\nTinyBERT scatter plot saved to {plot_path}")
    except Exception as e:
        logging.error(f"Failed to generate TinyBERT scatter plot: {str(e)}")
        print(f"\nError: Failed to generate TinyBERT scatter plot: {str(e)}")
        raise

def evaluate_model(model, dataloader, device, topic_to_label):
    model.eval()
    predictions = []
    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            outputs = model(**inputs, output_hidden_states=True)
            probs = torch.softmax(outputs.logits, dim=-1)
            conf, preds = torch.max(probs, dim=-1)
            for text, pred, conf_value, topic_probs in zip(batch['text'], preds, conf, batch['topic_probs']):
                predictions.append({
                    'processed_reviews': text,
                    'predicted_label': pred.item(),
                    'confidence': conf_value.item(),
                    'topic_probs': ','.join(map(str, topic_probs.numpy()))
                })
            embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            all_embeddings.extend(embeddings)
    
    predictions_df = pd.DataFrame(predictions)
    label_to_topic = {v: k for k, v in topic_to_label.items()}
    predictions_df['topic_name'] = predictions_df['predicted_label'].map(label_to_topic)
    
    all_embeddings = np.array(all_embeddings)
    logging.info(f"Embeddings shape before normalization: {all_embeddings.shape}")
    all_embeddings = normalize(all_embeddings, norm='l2')
    logging.info("Embeddings normalized for silhouette score computation")
    
    logging.info(f"Embeddings shape for silhouette score: {all_embeddings.shape}")
    if len(np.unique(predictions_df['predicted_label'])) > 1 and len(all_embeddings) > 1:
        sil_score = silhouette_score(all_embeddings, predictions_df['predicted_label'], metric='cosine')
        logging.info(f"Silhouette Score: {sil_score:.4f}")
        print(f"\nSilhouette Score: {sil_score:.4f}")
    else:
        logging.warning("Silhouette Score not computed: Need at least 2 unique labels and samples")
        print("\nWarning: Silhouette Score not computed due to insufficient unique labels or samples")
        sil_score = None
    
    return predictions_df, all_embeddings, sil_score

def check_dependencies():
    try:
        import umap
        import matplotlib
        import seaborn
        import sklearn
        logging.info("All dependencies (umap-learn, matplotlib, seaborn, scikit-learn) are installed")
    except ImportError as e:
        logging.error(f"Missing dependency: {str(e)}")
        print(f"\nError: Missing dependency: {str(e)}")
        raise

def main():
    setup_logging()
    logging.info("Checking dependencies...")
    check_dependencies()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    if device.type == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description='TinyBERT Inference on Test Data')
    parser.add_argument('--test_data', type=str, default='../../datasets/review_test.csv',
                        help='Path to test data CSV')
    parser.add_argument('--model_dir', type=str, default='../../train/BERT/saved_models/final_model',
                        help='Directory of saved model and tokenizer')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--output_dir', type=str, default='../../datasets',
                        help='Directory to save predictions')
    parser.add_argument('--plot_dir', type=str, default='embeddings_plots',
                        help='Directory to save embedding plot')
    args = parser.parse_args()

    try:
        os.makedirs(args.plot_dir, exist_ok=True)
        logging.info(f"Ensured plot directory exists: {args.plot_dir}")
    except Exception as e:
        logging.error(f"Failed to create plot directory {args.plot_dir}: {str(e)}")
        print(f"\nError: Failed to create plot directory {args.plot_dir}: {str(e)}")
        raise

    logging.info("Loading test data...")
    try:
        test_df = pd.read_csv(args.test_data)
        if 'processed_reviews' not in test_df.columns:
            raise ValueError("Test CSV must contain 'processed_reviews' column")
        test_df['processed_reviews'] = test_df['processed_reviews'].fillna("")
        
        # Filter out empty reviews
        valid_indices = [i for i, text in enumerate(test_df['processed_reviews']) if text.strip()]
        test_df = test_df.iloc[valid_indices].copy()
        logging.info(f"Filtered to {len(test_df)} valid reviews")
        
        test_df['topic_probs'] = [np.zeros(7)] * len(test_df)  # Assuming 7 topics
        topic_to_label = {'F': 0, 'FT': 1, 'PE': 2, 'PO': 3, 'SC': 4, 'SE': 5, 'US': 6}
        logging.info(f"Using topic-to-label mapping: {topic_to_label}")
        print(f"\nUsing topic-to-label mapping: {topic_to_label}")
    except Exception as e:
        logging.error(f"Failed to load test data {args.test_data}: {str(e)}")
        print(f"\nError: Failed to load test data {args.test_data}: {str(e)}")
        raise

    logging.info("Loading model and tokenizer...")
    try:
        tokenizer = BertTokenizer.from_pretrained(args.model_dir)
        model = BertForSequenceClassification.from_pretrained(args.model_dir).to(device)
        logging.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model or tokenizer: {str(e)}")
        print(f"\nError: Failed to load model or tokenizer: {str(e)}")
        raise

    test_loader = DataLoader(
        ReviewDataset(test_df['processed_reviews'].values, test_df['topic_probs'].values, tokenizer),
        batch_size=args.batch_size,
        pin_memory=True
    )

    logging.info("Running inference...")
    predictions_df, embeddings, sil_score = evaluate_model(model, test_loader, device, topic_to_label)
    output_path = os.path.join(args.output_dir, 'test_predictions.csv')
    
    try:
        predictions_df[['processed_reviews', 'predicted_label', 'topic_name', 'confidence', 'topic_probs']].to_csv(output_path, index=False)
        logging.info(f"Predictions saved to {output_path}")
        print(f"\nPredictions saved to {output_path}")
        logging.info("Sample of predictions:")
        logging.info(predictions_df[['processed_reviews', 'topic_name', 'confidence']].head(2).to_string())
        print("\nSample of predictions:")
        print(predictions_df[['processed_reviews', 'topic_name', 'confidence']].head(2))
        print("\nFor manual comparison, use the predictions in {} with LDA topics in new_reviews_with_topic.csv".format(output_path))
    except PermissionError as e:
        logging.error(f"Failed to save predictions to {output_path}: {e}")
        print(f"\nError: Failed to save predictions to {output_path}: {e}")
        raise

    try:
        logging.info("Generating embedding visualization...")
        plot_embeddings(embeddings, predictions_df['predicted_label'], list(topic_to_label.keys()), args.plot_dir)
    except Exception as e:
        logging.error(f"Failed to generate embedding visualization: {str(e)}")
        print(f"\nError: Failed to generate embedding visualization: {str(e)}")
        raise

    logging.info("Inference completed successfully")

if __name__ == "__main__":
    main()
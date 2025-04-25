#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from tqdm import tqdm
import argparse

def main():
    # Download NLTK data
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

    # Seed words definition
    seed_words = {
        'F': ['clinical', 'lead', 'view', 'player', 'class', 'data', 'allow', 'dispute', 'part', 'list', 'include', 'program', 'administrator', 'case', 'cohort', 'staff', 'member', 'student', 'site', 'lab', 'meeting', 'display', 'search', 'information', 'section', 'game', 'request', 'nursing', 'able'],
        'US': ['estimator', 'help', 'training', 'successfully', 'intuitive', 'click', 'realtor', 'using', 'collision', 'screen', 'first'],
        'SE': ['secure', 'stored', 'allowed', 'secured', 'log', 'encrypted', 'authorized', 'policy', 'security', 'role', 'company', 'ensure', 'password', 'supervisor', 'login'],
        'PO': ['wide', 'software', 'reasonable', 'portable', 'compatible', 'since', 'several', 'major', 'platform', 'palm', 'unix', 'io', 'android', 'fully', 'function', 'mobile', 'way', 'may', 'either'],
        'A': ['long', 'achieve', 'period', 'must', 'service', 'application', 'year', 'day', 'available', 'support', 'internet', 'time', 'use', 'customer', 'product', 'contractual', 'wcs', 'hour', 'availability', 'system', 'technical', 'shall', 'per', 'website', 'access', 'user', 'provide', 'online', 'schedule', 'uptime']
    }
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        # Lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = text.split()
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
                  and word not in ['course', 'learn', 'good', 'great']]
        return tokens

    # Argument parsing
    parser = argparse.ArgumentParser(description='Run seeded LDA topic modeling')
    parser.add_argument('--input', default='../datasets/preprocessed_coursera_review.csv', help='Input CSV file path')
    parser.add_argument('--output', default='../datasets/selected_pseudo_labeled_reviews.csv', help='Output CSV file path')
    parser.add_argument('--threshold', type=float, default=0.8, help='Confidence threshold for topic assignment')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    df = pd.read_csv(args.input)
    df = df[['processed_reviews']]  # Keep only the review column
    df['processed_reviews'] = df['processed_reviews'].fillna("")

    # Apply preprocessing with progress bar
    print("Preprocessing reviews...")
    tqdm.pandas()
    tokenized_reviews = df['processed_reviews'].progress_apply(preprocess)

    # Create dictionary
    dictionary = corpora.Dictionary(tokenized_reviews)

    # Filter extreme words from dictionary
    dictionary.filter_extremes(no_below=5, no_above=0.8)

    # Convert seed words to IDs
    seed_word_ids = {
        topic: [dictionary.token2id[word] for word in words if word in dictionary.token2id]
        for topic, words in seed_words.items()
    }

    # Map topic names to integer indices
    topic_name_to_id = {name: idx for idx, name in enumerate(seed_words.keys())}

    # Initialize eta with default prior
    num_topics = len(seed_words)
    num_words = len(dictionary)
    eta = np.ones((num_topics, num_words)) * 0.01

    # Assign strong priors to seed words
    for topic_name, seed_word_ids_in_topic in seed_word_ids.items():
        topic_id = topic_name_to_id[topic_name]
        for word_id in seed_word_ids_in_topic:
            eta[topic_id][word_id] = 100.0

    # Convert tokenized reviews to BoW corpus
    corpus = [dictionary.doc2bow(text) for text in tokenized_reviews]

    # Train the model
    print("Training LDA model...")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=20,
        alpha='asymmetric',
        iterations=400,
        eta=eta,
        random_state=42,
        minimum_probability=0.05,
        per_word_topics=True,
        decay=0.7,
        offset=50.0
    )

    # Show topics
    print("\nDiscovered Topics:")
    for topic in lda_model.print_topics(num_words=10):
        print(topic)

    # Compute coherence score
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=tokenized_reviews,
        dictionary=dictionary,
        coherence='c_v',
        topn=10
    )
    print(f"\nCoherence Score: {coherence_model.get_coherence():.4f}")

    # Get topic distribution for each document
    doc_topics = [lda_model.get_document_topics(doc) for doc in corpus]

    # Convert to numpy array (matrix: num_docs x num_topics)
    topic_matrix = np.zeros((len(corpus), num_topics))

    for i, topics in enumerate(doc_topics):
        for topic_id, prob in topics:
            topic_matrix[i, topic_id] = prob

    # FIRST SAVE: Save all reviews with Topic and topic_name (exactly as specified)
    all_topics_df = df.copy()
    all_topics_df['Topic'] = np.argmax(topic_matrix, axis=1)
    all_topics_df['topic_name'] = all_topics_df['Topic'].map({v: k for k, v in topic_name_to_id.items()})
    
    # Save to CSV with exactly the specified columns
    all_topics_path = '../datasets/coursera_reviews_with_topic_names.csv'
    all_topics_df[['processed_reviews', 'Topic', 'topic_name']].to_csv(all_topics_path, index=False)
    print(f"\nAll reviews with topic assignments saved to {all_topics_path}")
    print("Sample of saved data:")
    print(all_topics_df[['processed_reviews', 'Topic', 'topic_name']].head(2))

    # SECOND SAVE: Select high-confidence pseudo-labeled reviews (exactly as specified)
    selected_reviews = {topic_name: [] for topic_name in seed_words.keys()}
    id_to_topic = {v: k for k, v in topic_name_to_id.items()}

    print(f"\nSelecting reviews with confidence > {args.threshold}...")
    for i, (review_text, topic_probs) in enumerate(zip(df['processed_reviews'], topic_matrix)):
        dominant_topic = np.argmax(topic_probs)
        dominant_prob = topic_probs[dominant_topic]
        
        if dominant_prob > args.threshold:
            topic_name = id_to_topic[dominant_topic]
            selected_reviews[topic_name].append({
                'original_index': i,
                'topic': topic_name,
                'confidence': dominant_prob,
                'text': review_text
            })

    # Print summary
    print("\nSelected Reviews Summary:")
    for topic_name, reviews in selected_reviews.items():
        print(f"Topic {topic_name}: {len(reviews):,} reviews selected")

    # Create DataFrame with exactly the specified columns
    selected_data = []
    for topic_name, reviews in selected_reviews.items():
        for rev in reviews:
            selected_data.append({
                'original_index': rev['original_index'],
                'topic': rev['topic'],
                'confidence': rev['confidence'],
                'text': rev['text']
            })

    selected_df = pd.DataFrame(selected_data)

    # Save to CSV with exactly the specified columns
    selected_path = '../datasets/selected_pseudo_labeled_reviews.csv'
    selected_df.to_csv(selected_path, index=False)
    print(f"\nHigh-confidence pseudo-labeled reviews saved to {selected_path}")
    print("Sample of saved data:")
    print(selected_df.head(2))

if __name__ == "__main__":
    main()
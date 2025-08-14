Of course! This Python script performs a sophisticated natural language processing task called **Seeded Latent Dirichlet Allocation (Seeded LDA)**. Its main goal is to automatically categorize text reviews into predefined topics.

Here’s a breakdown of what the code does, step by step.

### High-Level Goal

Imagine you have thousands of user reviews and you want to sort them into specific categories like "Security," "Performance," or "Usability." Doing this manually would be incredibly slow.

This script automates the process. It uses a machine learning model (LDA) that learns the underlying topics in the text. Crucially, it's "seeded," meaning we give the model hints by providing a list of words we expect to see in each category. This makes the model's topics more accurate and aligned with our definitions.

Finally, it identifies reviews that the model is very confident about and saves them as a "pseudo-labeled" dataset, which can be used for training other machine learning models.

---

### ## Code Explanation

---

### **1. Import Libraries and Configure Logging**
The code starts by importing a suite of powerful Python libraries, each serving a specific role in the text analysis pipeline:

- **Pandas and NumPy**: These are the workhorses for data manipulation (`pandas`) and numerical computations (`numpy`). Pandas handles data frames (like spreadsheets), while NumPy deals with arrays and mathematical operations.
- **NLTK (Natural Language Toolkit)**: Provides tools for text preprocessing, like tokenization (splitting text into words), stopword removal, and lemmatization (reducing words to their root form, e.g., "running" to "run").
- **Gensim**: A library for topic modeling and document similarity analysis. It’s used here for LDA modeling, creating dictionaries, and computing coherence scores.
- **Scikit-learn’s TfidfVectorizer**: Converts text into numerical features based on Term Frequency-Inverse Document Frequency (TF-IDF), which highlights important words in a document.
- **RAKE (Rapid Automatic Keyword Extraction)**: Extracts key phrases from text, useful for identifying significant terms.
- **Other utilities**: `re` for regular expressions, `Counter` for counting word frequencies, `tqdm` for progress bars, `logging` for tracking the process, `random` for sampling, `os` for file handling, and `scipy.stats.entropy` for measuring topic distribution uncertainty.

**Logging Setup**: The script configures logging to save progress and errors to a file (`lda_output.log`). It sets the logging level to `INFO` for general updates and `WARNING` for the `gensim` library to reduce noise. The log format includes timestamps, making it easier to debug or track the process later.

**NLTK Data Download**: The script quietly downloads required NLTK resources (`punkt` for tokenization, `stopwords` for common words to ignore, and `wordnet` for lemmatization). This ensures the NLP tools are ready to go.

---

### **2. Initialize NLP Tools and Stopwords**
Here, the script sets up essential NLP tools and defines stopwords to filter out uninformative words.

- **Lemmatizer**: The `WordNetLemmatizer` reduces words to their base form (e.g., "cats" to "cat"), which helps standardize terms for analysis.
- **Stopwords**: The script starts with NLTK’s English stopwords (e.g., "the," "is") and adds two custom sets:
  - **Domain-specific stopwords**: Words like "system," "software," and "user" that may be common in the dataset but not topic-specific.
  - **Review-specific stopwords**: Words like "good," "great," or "helpful" that are common in reviews but don’t add topical insight.
  
These stopwords are combined into a single set to be excluded during text processing, ensuring the analysis focuses on meaningful terms.

---

### **3. Load PROMISE Data**
The `load_promise_data` function reads a CSV file (`PROMISE_exp_cleaned.csv`) into a Pandas DataFrame using `pd.read_csv`. It includes error handling:
- If the file isn’t found, it logs an error and raises a `FileNotFoundError`.
- For other issues (e.g., corrupted file), it logs the exception and raises it.

The function is called to load the dataset, and the script logs a success message with the file path. This dataset likely contains text data (e.g., requirements or reviews) labeled by categories in a column called `_class_`.

---

### **4. Aggregate Data by Category**
The script groups the DataFrame by the `_class_` column and concatenates all `cleaned_text` entries for each category into a single string using `apply(lambda x: ' '.join(x))`. This creates a dictionary (`category_texts`) where keys are category names and values are the aggregated texts.

For each category, it:
- Counts the number of words in the aggregated text.
- Logs and prints the word count.
- Issues warnings if the text is too short (<50 words) or empty (0 words), raising an error for empty texts to ensure data quality.

This step prepares the text for seed word extraction by consolidating all relevant text per category.

---

### **5. Extract Seed Words**
This section is the heart of generating *seed words*—terms that represent each category and guide the LDA model. The process is intricate, combining TF-IDF and RAKE for robust keyword extraction.

For each category’s text:
1. **TF-IDF Vectorization**:
   - A `TfidfVectorizer` is created with parameters:
     - `max_features=200`: Limits to the top 200 terms.
     - `ngram_range=(1, 3)`: Considers unigrams, bigrams, and trigrams.
     - `stop_words`: Excludes the custom stopwords.
     - `min_df=1`: Includes terms appearing in at least one document.
     - `sublinear_tf=True`: Applies sublinear scaling to TF-IDF scores to reduce the impact of frequent terms.
   - The vectorizer transforms the category’s text into a TF-IDF matrix, and terms with scores > 0.05 are retained.

2. **RAKE Keyword Extraction**:
   - RAKE extracts key phrases (2-3 words) using the `DEGREE_TO_FREQUENCY_RATIO` metric, ignoring stopwords and repeated phrases.
   - The top 60 phrases are selected, split into words, and lemmatized. Multi-word phrases are joined with underscores (e.g., "machine_learning").

3. **Combine and Filter Terms**:
   - TF-IDF terms and RAKE terms are combined into a unique set.
   - Terms are filtered to:
     - Be alphabetic or contain underscores (for multi-word phrases).
     - Not be in stopwords.
     - Have more than 2 characters (excluding underscores).
     - Have a TF-IDF score > 0.06 (for unigrams) or be a multi-word phrase.

4. **Score and Select Terms**:
   - Each term is scored using its TF-IDF score or, if unavailable, its normalized frequency (word count divided by total words).
   - The top 20 terms (based on `target_word_count`) are selected as seed words for the category.

The result is a dictionary (`seed_words`) mapping each category to its top 20 seed words, with scores stored in `category_word_scores`.

---

### **6. Deduplicate and Finalize Seed Words**
To ensure seed words are unique across categories (avoiding overlap), the script:
- Collects all terms with their category and score.
- Sorts them by score (highest first).
- Assigns each term to its highest-scoring category, storing the assignment in `final_seed_words` and tracking mappings in `word_to_category` and `word_to_score`.
- If a category has fewer than 20 words, it fills the remaining slots with high-scoring terms not yet assigned, ensuring each category has exactly 20 seed words.

---

### **7. Validate and Save Seed Words**
The script validates the seed words:
- Logs and prints the seed words for each category.
- Checks for overlaps (words appearing in multiple categories). If overlaps exist, it logs and prints warnings; otherwise, it confirms no overlaps.
- Saves the seed words to a CSV file (`seed_words.csv`) with columns `Category` and `SeedWord`.

It then reloads the seed words using `load_seed_words_from_csv` and removes them from the stopwords set to ensure they’re not filtered out later.

---

### **8. Define Preprocessing Function**
The `preprocess` function prepares text for LDA:
- Checks if the input is a valid, non-empty string.
- Converts text to lowercase.
- Removes non-alphabetic characters using a regular expression.
- Splits text into tokens (words).
- Lemmatizes tokens and removes stopwords and words with ≤2 characters.
- Returns a list of processed tokens or an empty list if invalid.

---

### **9. Load and Preprocess Review Data**
The script loads a review dataset (`review_train.csv`), keeping only the `processed_reviews` column and filling missing values with empty strings. It:
- Applies the `preprocess` function to each review using `tqdm` for a progress bar.
- Filters out reviews with no valid tokens, creating `filtered_df` with valid reviews only.
- Logs the number of valid reviews.

---

### **10. Detect Bigrams**
To capture meaningful phrases (e.g., "machine_learning"), the script:
- Samples up to 100,000 reviews to train a bigram model using `gensim.models.Phrases`.
- Sets `min_count=15` (phrases must appear 15+ times) and `threshold=0.7` with NPMI scoring for phrase quality.
- Applies the bigram model to all tokenized reviews, joining words into phrases where appropriate.

---

### **11. Create Dictionary and Check Seed Words**
The script creates a Gensim dictionary from tokenized reviews and filters extreme terms (appearing in <1 document or >80% of documents). It then checks:
- If seed words are present in the corpus (tokenized reviews).
- If seed words survive dictionary filtering.
- Logs and prints warnings for missing words, suggesting near-matches for those absent from the corpus.

---

### **12. Validate Seed Words and Prepare ETA**
The script maps seed words to dictionary IDs and creates an `eta` matrix to guide the LDA model:
- `eta` is initialized as a matrix of size (number of topics × dictionary size) with small values (0.01).
- For each topic’s seed words, their corresponding dictionary IDs get a high `eta` value (500.0), encouraging the LDA model to associate these words with the topic.

A `topic_name_to_id` mapping is created for reference, and the script logs the number of seed words per topic in the dictionary.

---

### **13. Train LDA Model**
The script trains the LDA model:
- Converts tokenized reviews to a Bag-of-Words (BoW) corpus using the dictionary.
- Trains the `LdaModel` with:
  - `num_topics`: Number of categories.
  - `passes=20`: Number of training iterations.
  - `alpha=0.01/num_topics`: Low alpha for sparse topic distributions.
  - `iterations=400`: Maximum iterations per document.
  - `eta`: Guides topic-word assignments.
  - `random_state=42`: Ensures reproducibility.
  - `minimum_probability=0.05`: Ignores low-probability topics.
  - `per_word_topics=True`: Tracks word-topic assignments.
  - `decay=0.7` and `offset=50.0`: Control learning rate.

The model, dictionary, and bigram phraser are saved to the `models/` directory.

---

### **14. Analyze Topics and Coherence**
The script evaluates the LDA model:
- Computes per-topic coherence (`c_v`) using `CoherenceModel` with a window size of 50 and top 10 words.
- Logs and prints coherence scores for each topic and the top 30 words per topic for inspection.
- Re-computes coherence for multiple `topn` values (10 to 100, step 10) to assess model robustness.

---

### **15. Verify Seed Word Probabilities**
For each seed word, the script checks its probability and rank in each topic, ensuring the LDA model aligns seed words with their intended topics. It logs and prints:
- The topic, probability, and rank for each seed word.
- Warnings for seed words not in the dictionary.

---

### **16. Assign Topics to Reviews and Save**
The script assigns topics to reviews:
- Gets topic distributions for each review using `lda_model.get_document_topics`.
- Creates a topic matrix where each row is a review and each column is a topic probability.
- Saves all reviews with their dominant topic and topic name to `reviews_with_topic.csv`.

It then selects high-confidence reviews (dominant topic probability > 0.8 and entropy ≤ 1.0) and saves them to `selected_pseudo_labeled.csv`. Non-high-confidence reviews are saved to `non_high_confidence_reviews.csv`. Entropy measures topic distribution uncertainty, filtering out ambiguous reviews.

---

### **Big Picture**
This script is a sophisticated pipeline for topic modeling on a categorized dataset (PROMISE) and a review dataset. It:
1. Extracts seed words per category using TF-IDF and RAKE.
2. Ensures seed word uniqueness and saves them.
3. Preprocesses reviews with lemmatization, stopword removal, and bigram detection.
4. Trains an LDA model guided by seed words.
5. Evaluates the model with coherence scores and topic inspections.
6. Assigns topics to reviews, filtering high-confidence ones for pseudo-labeling.

It’s designed for tasks like classifying reviews into predefined categories, with logging and validation to ensure robustness. If you have specific questions or want to dive deeper into any part, let me know!


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### **What is LDA, and What Does It Do?**
Latent Dirichlet Allocation (LDA) is a statistical model used in natural language processing to discover hidden (latent) topics in a collection of documents. Imagine you have a pile of reviews, and you want to figure out what themes or topics (e.g., “customer service,” “product quality”) are discussed without manually reading each one. LDA does this by analyzing word patterns and grouping words that frequently appear together into topics. Each document (e.g., a review) is then represented as a mix of these topics, and each topic is a mix of words with associated probabilities.

In the script, LDA is used to:
1. Identify topics in a review dataset (`review_train.csv`) based on seed words derived from a categorized dataset (`PROMISE_exp_cleaned.csv`).
2. Assign each review to a dominant topic with a confidence score, effectively classifying reviews into predefined categories.
3. Save high-confidence topic assignments for pseudo-labeling, which can be used for further analysis or machine learning tasks.

---

### **How LDA Works: The Core Idea**
LDA assumes that:
- **Documents are mixtures of topics**: Each review is a blend of multiple topics (e.g., 70% “customer service,” 20% “product quality,” 10% “pricing”).
- **Topics are mixtures of words**: Each topic is defined by a set of words with probabilities (e.g., the “customer service” topic might assign high probabilities to words like “support,” “response,” and “helpful”).
- **Words are generated probabilistically**: When a document is “written,” LDA imagines it picks a topic (based on the document’s topic distribution) and then picks a word from that topic’s word distribution.

The goal of LDA is to reverse-engineer this process: given the words in the documents, figure out the topics and their word distributions, as well as each document’s topic proportions.

---

### **LDA in the Script: Step-by-Step Mechanics**
Let’s walk through how the LDA model is implemented and works in the script, following the code’s flow and tying it to the broader LDA concept.

#### **1. Preparing the Data for LDA**
Before LDA can work, the script preprocesses the text to make it suitable for modeling:

- **Loading and Preprocessing Reviews**:
  - The script loads `review_train.csv`, which contains a column `processed_reviews` with text data.
  - The `preprocess` function cleans each review by:
    - Converting to lowercase.
    - Removing non-alphabetic characters (e.g., punctuation, numbers).
    - Splitting into words (tokens).
    - Lemmatizing (e.g., “running” to “run”) using `WordNetLemmatizer`.
    - Removing stopwords (common words like “the” and custom words like “good,” “helpful”) and short words (≤2 characters).
  - This produces `tokenized_reviews`, a list of lists where each inner list contains the processed tokens of a review (e.g., `["customer", "support", "fast"]`).

- **Detecting Bigrams**:
  - The script uses `gensim.models.Phrases` to identify common two-word phrases (bigrams) like “machine_learning” or “customer_service.”
  - It trains a bigram model on a sample of up to 100,000 reviews, requiring phrases to appear at least 15 times with a normalized pointwise mutual information (NPMI) score ≥ 0.7.
  - The `bigram_phraser` applies these bigrams to all reviews, so tokens like `["machine", "learning"]` become `["machine_learning"]`. This helps capture multi-word concepts as single units.

- **Creating a Dictionary and Corpus**:
  - A `gensim.corpora.Dictionary` is created from `tokenized_reviews`, mapping each unique word (or bigram) to an ID.
  - The dictionary is filtered to remove words appearing in fewer than 1 document or more than 80% of documents (`no_below=1`, `no_above=0.8`), focusing on moderately frequent, discriminative terms.
  - The corpus is created as a Bag-of-Words (BoW) representation using `dictionary.doc2bow`, where each review is a list of (word ID, frequency) pairs (e.g., `[(0, 2), (5, 1)]` means word ID 0 appears twice, word ID 5 appears once).

- **Seed Words and ETA Matrix**:
  - The script extracts seed words from the PROMISE dataset, which are representative terms for each category (e.g., “support” for a “customer service” category). These are stored in `seed_words`, a dictionary mapping categories to lists of 20 terms.
  - Seed words are mapped to dictionary IDs, and an `eta` matrix is created (size: number of topics × dictionary size). Most entries are set to 0.01, but seed word IDs for each topic get a high value (500.0), guiding LDA to associate these words with their respective topics.

#### **2. Training the LDA Model**
The script trains the LDA model using `gensim.models.LdaModel`. Here’s how it’s configured and what it does:

- **Key Parameters**:
  - `corpus`: The BoW corpus of reviews.
  - `id2word`: The dictionary mapping word IDs to words.
  - `num_topics`: Set to the number of categories in `seed_words` (derived from the PROMISE dataset’s `_class_` column).
  - `passes=20`: The model iterates over the entire corpus 20 times to refine topic assignments.
  - `iterations=400`: For each document, the model performs up to 400 iterations to optimize topic proportions.
  - `alpha=0.01/num_topics`: A low alpha encourages each review to focus on a few dominant topics (sparse topic distribution).
  - `eta`: The matrix guiding topic-word assignments, with high values for seed words to align topics with categories.
  - `random_state=42`: Ensures reproducible results.
  - `minimum_probability=0.05`: Ignores topic probabilities below 0.05 for cleaner output.
  - `per_word_topics=True`: Tracks which topics each word is assigned to.
  - `decay=0.7` and `offset=50.0`: Control the learning rate, balancing new and old evidence during training.

- **What LDA Does During Training**:
  - **Initialization**: LDA starts with random topic assignments for each word in each review.
  - **Iterative Updates**: For each pass:
    - For each review, LDA updates the topic proportions (how much of each topic the review contains) based on the words it has and the `alpha` prior.
    - For each word, LDA updates the topic assignment based on the `eta` matrix and how often the word appears with other words in the same topic across the corpus.
  - The `eta` matrix biases the model to assign seed words to their intended topics, aligning topics with the PROMISE dataset’s categories.
  - Over 20 passes, the model refines these assignments, converging to stable topic-word and document-topic distributions.

- **Output**: After training, the LDA model provides:
  - **Topic-word distributions**: For each topic, a list of words with probabilities (e.g., Topic 0: “support” 0.1, “response” 0.08, …).
  - **Document-topic distributions**: For each review, a list of (topic ID, probability) pairs (e.g., Review 1: Topic 0 0.7, Topic 1 0.2, Topic 2 0.1).

#### **3. How the Model Processes Data**
- **Input**: The BoW corpus, where each review is a sparse list of word IDs and their counts.
- **Processing**:
  - LDA uses the Dirichlet distribution to model topic proportions in documents and word proportions in topics.
  - The `eta` matrix ensures that seed words (e.g., “support” for a customer service topic) have high probabilities in their assigned topics.
  - The model iteratively adjusts topic assignments to maximize the likelihood that the observed words were generated by the inferred topics.
- **Output**: For each review, LDA assigns probabilities across all topics, indicating how much each topic contributes to the review. For example, a review with “fast response, helpful support” might be 80% Topic 0 (customer service) and 20% Topic 1 (product quality).

#### **4. Post-Training: Topic Assignment and Evaluation**
After training, the script uses the LDA model to:
- **Assign Topics**:
  - For each review, `lda_model.get_document_topics` returns a list of (topic ID, probability) pairs.
  - The script creates a topic matrix where each row is a review and each column is a topic’s probability.
  - The dominant topic (highest probability) is assigned as the review’s topic, mapped to a category name using `topic_name_to_id`.

- **Filter High-Confidence Reviews**:
  - Reviews with a dominant topic probability > 0.8 and topic entropy ≤ 1.0 are selected as high-confidence. Entropy measures uncertainty in the topic distribution (lower entropy means clearer topic assignment).
  - These reviews are saved to `selected_pseudo_labeled.csv` with their topic, confidence, and topic probabilities.
  - Non-high-confidence reviews are saved separately.

- **Evaluate Topics**:
  - The script computes `c_v` coherence scores to assess topic quality, using `CoherenceModel`. Coherence measures how well the top words in a topic co-occur in the corpus, with higher scores indicating more interpretable topics.
  - It checks coherence for different numbers of top words (10 to 100) to ensure robustness.
  - It verifies that seed words have high probabilities and ranks in their assigned topics, confirming alignment with the PROMISE categories.

#### **5. Saving the Model**
The trained LDA model, dictionary, and bigram phraser are saved to the `models/` directory for future use (e.g., applying the model to new reviews).

---

### **How the Model Fits into the Script’s Goal**
The LDA model is the core engine for classifying reviews into categories defined by the PROMISE dataset. Here’s how it ties everything together:
- **Seed Words as Guidance**: The script extracts seed words from the PROMISE dataset’s categories using TF-IDF and RAKE, ensuring each topic aligns with a category (e.g., “customer service” or “product quality”).
- **Preprocessing for Quality**: Text cleaning, stopword removal, and bigram detection ensure the model focuses on meaningful terms.
- **Guided LDA**: The `eta` matrix biases the model toward seed words, making topics interpretable and aligned with predefined categories.
- **Pseudo-Labeling**: By selecting high-confidence reviews (high probability, low entropy), the script creates a labeled dataset for downstream tasks, like training a classifier or analyzing trends.
- **Evaluation**: Coherence scores and seed word verification ensure the topics are meaningful and correctly aligned.

---

### **A Simple Analogy**
Think of LDA as a librarian organizing a messy library of reviews. Each review is a book with a mix of topics (like chapters on “customer service” or “pricing”). The librarian:
1. Reads the books (corpus) and notices which words appear together often (e.g., “support” and “response”).
2. Uses a guide (seed words via `eta`) to label shelves with categories like “Customer Service.”
3. Assigns each book to shelves based on its content, allowing a book to sit on multiple shelves (topic proportions).
4. Picks the best-fitting shelf for each book (dominant topic) and checks if the assignment makes sense (coherence, entropy).

In the script, the PROMISE dataset provides the initial shelf labels (seed words), and LDA organizes the reviews onto these shelves, ensuring clarity and alignment.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

> Can you explain to me the parameter we use on the lda model and why we chose those number?

Of course. Explaining the parameters is key to understanding how you can control the model's behavior. The choices for these numbers are a deliberate balance between **model quality**, **training time**, and **interpretability of the results**.

Think of these parameters as the control knobs on a sophisticated machine. Let's go through them one by one, based on your script.

```python
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    passes=20,
    alpha=0.1/num_topics,
    iterations=400,
    eta=eta,
    random_state=42,
    minimum_probability=0.05,
    per_word_topics=True,
    decay=0.7,
    offset=50.0
)
```

### Core & Seeding Parameters

These parameters define the fundamental structure of the model and its inputs.

| Parameter | Value in Script | Why This Value Was Chosen |
| :---      | :---            | :---                      |
| `corpus`  | `corpus` | **This isn't a number, but the main data input.** It's the entire collection of reviews, converted into a Bag-of-Words numerical format that the model can understand. |
| `id2word` | `dictionary` | **This is the model's vocabulary.** It maps the numeric IDs used in the `corpus` back to actual words (e.g., ID 55 -\> "login"). It's essential for interpreting the results. |
| `num_topics` | `len(seed_words)` | **Reason:** In a **Seeded LDA**, the number of topics is not something we guess. It's pre-defined by us. Since we provided 7 sets of seed words (`F`, `FT`, `PE`, etc.), we are explicitly telling the model to find exactly 7 topics that correspond to our seeds. |
| `eta` | `eta` (the matrix) | **Reason:** This is the most important parameter for **Seeded** LDA. This matrix provides the "strong magnetic pull" we discussed. By setting high values for our seed words in their respective topics, we are **forcing the model's output to align with our predefined categories**. Without this, the model would be completely unsupervised. |

-----

### Training & Quality Control Parameters

These knobs control how the model learns and how stable and reliable the final topics are.

| Parameter | Value in Script | Why This Value Was Chosen |
| :--- | :--- | :--- |
| `passes` | `20` | **What it is:** The number of times the model trains on the entire dataset (`corpus`).\<br\>\<br\>**Reason:** `20` is a relatively high number. A single pass is often not enough for the topic assignments to stabilize. By iterating over the data 20 times, we give the model enough opportunity to converge on high-quality, stable topics. \<br\>*Trade-off:* More passes lead to better topics but significantly increase training time. `20` is a good choice for a final, production-quality model. |
| `iterations` | `400` | **What it is:** The maximum number of times the model will iterate over each individual document within a single pass.\<br\>\<br\>**Reason:** `400` is also a high number, aimed at ensuring quality. For each document, the model tries to find the best topic mixture. This parameter ensures that this process is done thoroughly for every document in every pass. The combination of high `passes` and high `iterations` prioritizes model quality over speed. |
| `alpha` | `0.1/num_topics` | **What it is:** A parameter that controls the expected topic distribution for documents. A high `alpha` assumes documents are made of many topics. A low `alpha` assumes documents are made of just a few topics.\<br\>\<br\>**Reason:** The script uses a very low, symmetric `alpha`. This encourages **topic sparsity**. We are telling the model that we expect most reviews to be about one or two main topics, not a little bit of everything. This makes the results much easier to interpret and is usually a more realistic assumption. |
| `random_state` | `42` | **What it is:** A seed for the model's random number generator.\<br\>\<br\>**Reason:** This is **critical for reproducibility**. LDA has a random initialization step. Without `random_state`, running the script twice on the same data would give you slightly different results. By setting it to `42` (or any integer), we guarantee that the model will produce the exact same output every single time, which is essential for consistent experiments. |

-----

### Filtering & Advanced Parameters

These fine-tune the model's output and learning process.

| Parameter | Value in Script | Why This Value Was Chosen |
| :--- | :--- | :--- |
| `minimum_probability` | `0.05` | **What it is:** A filter to remove topics from a document's results if their probability is below this threshold.\<br\>\<br\>**Reason:** This cleans up the output. If a document is 95% about "Security" and 0.1% about "Performance", we probably don't care about that tiny 0.1%. This setting tells the model to only report topic probabilities that are 5% or higher, making the final topic distribution for each document less noisy. |
| `decay` & `offset` | `0.7` & `50.0`| **What they are:** Advanced parameters that control the learning rate, especially in online training. \<br\>\<br\>**Reason:** While this script performs batch (not online) training, these parameters still influence how the learning rate is adjusted during iterations. The chosen values (`0.7` and `50.0`) are common and robust defaults from the original `gensim` research and provide a stable learning schedule. For most use cases, it's best to leave these at their recommended values unless you are doing advanced model tuning. |
| `per_word_topics` | `True` | **What it is:** Tells the model to keep track of the topic assignments for each individual word.\<br\>\<br\>**Reason:** This is required to calculate certain evaluation metrics, most importantly the **Coherence Score (`c_v`)**. The script calculates this score to grade the model's topic quality, so this parameter must be set to `True`. The only downside is that it uses more memory during training. |


1. Inisialisasi Acak
Apa yang terjadi? LDA memulai dengan "menebak" secara acak topik untuk setiap kata di dokumen. Misalnya, kata "nasi" bisa dianggap bagian dari topik 1, dan "goreng" di topik 2, tanpa aturan awal.
Parameter yang Berperan:

num_topics: Jumlah topik yang Anda tentukan (misalnya, 2 topik). Ini menentukan berapa banyak "kelompok" topik yang LDA coba buat. Jika Anda set 2 topik, setiap kata akan diassign secara acak ke salah satu dari dua topik ini.
random_state: Mengatur pola acak agar hasilnya konsisten jika dijalankan ulang. Tanpa ini, inisialisasi acak bisa berbeda setiap kali.


Sederhananya: Bayangkan Anda punya sekotak kelereng (kata-kata) dan ingin membaginya ke beberapa kotak (topik). num_topics menentukan jumlah kotak, dan random_state memastikan cara Anda membagi kelereng selalu sama jika ulang.

2. Iterasi

Apa yang terjadi? LDA melihat pola kemunculan kata (misalnya, "nasi" dan "goreng" sering muncul bersama) dan memperbarui topik untuk setiap kata. Kata-kata yang sering muncul bersama cenderung masuk ke topik yang sama.
Parameter yang Berperan:

passes: Jumlah berapa kali LDA memeriksa seluruh dokumen untuk memperbarui topik. Semakin banyak passes, semakin teliti model belajar, tapi butuh waktu lebih lama.
iterations: Jumlah iterasi dalam satu pass untuk memperbaiki distribusi topik. Ini seperti berapa kali LDA "menyempurnakan" tebakannya di setiap putaran.
alpha: Mengatur seberapa merata topik tersebar di dokumen. Jika alpha rendah, setiap dokumen cenderung punya sedikit topik (misalnya, hanya topik 1). Jika tinggi, dokumen bisa punya campuran banyak topik.
eta: Mengatur seberapa merata kata-kata tersebar di topik. Jika eta rendah, setiap topik punya sedikit kata kunci. Jika tinggi, topik bisa punya lebih banyak kata.


Sederhananya: Bayangkan Anda menyortir kelereng berulang-ulang. passes dan iterations seperti berapa kali Anda memeriksa dan menyusun ulang kelereng. alpha menentukan apakah setiap dokumen hanya punya satu jenis kelereng atau campuran. eta menentukan apakah setiap kotak topik punya sedikit atau banyak jenis kelereng.

3. Konvergensi

Apa yang terjadi? Setelah beberapa iterasi, LDA menemukan distribusi topik yang stabil, di mana kata-kata sudah dikelompokkan ke topik yang masuk akal, dan dokumen punya proporsi topik yang jelas.
Parameter yang Berperan:

passes dan iterations: Keduanya memastikan model punya cukup waktu untuk menemukan pola yang stabil. Jika terlalu sedikit, hasil bisa kurang akurat.
alpha dan eta: Memengaruhi hasil akhir. Misalnya, alpha tinggi membuat dokumen punya campuran topik yang lebih merata, sedangkan eta tinggi membuat topik punya lebih banyak kata yang relevan.


Sederhananya: Ini seperti saat kelereng sudah tersusun rapi di kotak yang tepat setelah beberapa kali sortir. passes dan iterations memastikan Anda sortir cukup lama, sedangkan alpha dan eta memastikan kelereng tersebar dengan cara yang logis.
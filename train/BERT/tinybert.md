This script trains a BERT-based model for text classification, incorporating pseudo-labeling and a topic-informed loss function to enhance performance on review data. It leverages the pseudo-labeled output from the previous LDA script (`selected_pseudo_labeled.csv`) and unlabeled reviews (`non_high_confidence_reviews.csv`) to fine-tune a TinyBERT model, visualize embeddings, and evaluate classification performance. I’ll break it down section by section, narrating what each part does, why it’s important, and how it fits into the bigger picture, while keeping the explanation thorough and accessible.

---

### **1. Shebang and Initial Setup**
The script starts with a shebang (`#!/usr/bin/env python3`), ensuring it runs with Python 3 on Unix-like systems. The commented-out line `os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'` suggests TensorFlow might have been used in an earlier version but is currently unused, as the script relies on PyTorch.

**Libraries Imported**:
- **PyTorch (`torch`)**: The core framework for building and training the neural network, handling tensors, and leveraging GPU acceleration.
- **Transformers (`BertForSequenceClassification`, `BertTokenizer`)**: From Hugging Face, these provide a pre-trained BERT model and tokenizer for text processing.
- **Pandas and NumPy**: For data manipulation and numerical operations.
- **Scikit-learn**: For metrics (`accuracy_score`, `precision_recall_fscore_support`) and splitting data (`train_test_split`).
- **Tqdm**: For progress bars during training and evaluation.
- **Argparse**: To handle command-line arguments for flexible configuration.
- **Logging**: To record progress and results to both file and console.
- **Matplotlib**: For plotting embeddings.
- **Shutil**: For file operations like deleting directories.
- **UMAP**: For dimensionality reduction to visualize high-dimensional embeddings.
- **Datetime**: For timestamping logs and saved models.

**GPU Check**:
The script checks for GPU availability using `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` and prints the device (e.g., “cuda” or “cpu”) and GPU name if applicable. This ensures the model can leverage GPU acceleration for faster training.

---

### **2. Utility Functions**
The script defines several helper functions to manage logging, saving, visualization, and loss computation.

- **setup_logging**:
  - Configures logging to save messages to a file (named with a timestamp, e.g., `training_log_20250730_133000.txt`) and print to the console.
  - Uses `logging.INFO` level for detailed updates, with a format including timestamp, level, and message.

- **save_model**:
  - Saves the BERT model and tokenizer to a specified directory (`output_dir`).
  - Deletes any existing directory to avoid conflicts, creates a new one, and saves the model using Hugging Face’s `save_pretrained`.
  - Writes metadata (iteration and timestamp) to a `training_info.txt` file.
  - Logs the save operation.

- **plot_embeddings**:
  - Uses UMAP to reduce high-dimensional BERT embeddings (from the `[CLS]` token) to 2D for visualization.
  - Creates a scatter plot colored by class labels, saved as a PNG file (e.g., `embedding_iter1_epoch1.png`) in the `embeddings_plots` directory.
  - Helps visualize how well the model separates classes in the embedding space.

- **topic_informed_loss**:
  - Computes a custom loss combining:
    - **Cross-Entropy Loss**: Measures the difference between predicted logits and true labels, weighted by `class_weights` to handle class imbalance.
    - **KL Divergence Loss**: Encourages the model’s predicted probabilities to align with topic probabilities (from the LDA script), using `torch.nn.KLDivLoss`.
  - Combines them with a weighted sum: `alpha * ce_loss + (1 - alpha) * soft_loss`, where `alpha=0.7` prioritizes cross-entropy but incorporates topic information.
  - This loss leverages LDA topic probabilities to guide the model, enhancing alignment with predefined topics.

---

### **3. Custom Dataset Class**
The `ReviewDataset` class extends `torch.utils.data.Dataset` to prepare data for training:
- **Initialization**:
  - Takes `texts` (review texts), `labels` (class labels), `topic_probs` (LDA topic probabilities), `tokenizer` (BERT tokenizer), and `max_length` (default 128 tokens).
- **Methods**:
  - `__len__`: Returns the number of reviews.
  - `__getitem__`: For each review:
    - Encodes the text using the BERT tokenizer, padding/truncating to `max_length`, and returning `input_ids` and `attention_mask` as tensors.
    - Includes the label as a `torch.long` tensor and topic probabilities as a `torch.float` tensor.
    - Returns the original text for pseudo-labeling.
  - The output is a dictionary suitable for PyTorch’s `DataLoader`.

---

### **4. Training and Evaluation Functions**
- **train_model**:
  - Trains the model for one epoch using the `topic_informed_loss`.
  - Iterates over batches in the `dataloader`, moving inputs and labels to the device (GPU/CPU).
  - Computes logits, calculates loss, backpropagates gradients, and updates model parameters using the optimizer and scheduler.
  - Returns the average loss per batch.

- **evaluate_classification_metrics**:
  - Evaluates the model on a validation set, computing:
    - **Accuracy**, **precision**, **recall**, and **weighted F1 score** using scikit-learn.
    - **Per-class F1 scores** for each topic, logged with topic names.
  - Extracts `[CLS]` token embeddings from the last hidden layer for visualization.
  - Calls `plot_embeddings` to save a UMAP plot.
  - Returns a dictionary with metrics.

- **generate_pseudo_labels**:
  - Uses the model to predict labels for unlabeled reviews.
  - For each batch, computes softmax probabilities, selects predictions with confidence > `threshold` (default 0.8), and creates one-hot topic probabilities.
  - Returns a DataFrame with high-confidence pseudo-labels (`text`, `label`, `confidence`, `topic_probs`).
  - Checks overlap with existing LDA topics, logging the percentage of matching predictions.

- **check_data_leakage**:
  - Ensures no text overlap between training and validation sets to prevent overfitting.
  - Logs a warning if duplicates are found, otherwise confirms no leakage.

---

### **5. Main Function**
The `main` function orchestrates the entire process:

#### **Setup**:
- Calls `setup_logging` and re-checks GPU availability, enabling cuDNN benchmarking for optimized GPU performance.
- Parses command-line arguments for:
  - Paths to pseudo-labeled (`selected_pseudo_labeled.csv`) and unlabeled data (`non_high_confidence_reviews.csv`).
  - Hyperparameters: `batch_size` (8), `epochs` (3), `learning_rate` (5e-5), `warmup_steps` (100), `threshold` (0.8), `output_dir` (`saved_models`).

#### **Data Loading**:
- Loads the pseudo-labeled data, removes duplicates, and maps topics to integer labels (`topic_to_label`).
- Parses topic probabilities from comma-separated strings to NumPy arrays.
- Computes **class weights** to handle imbalance: for each class, weight = `N / (count * C)`, where `N` is the dataset size and `C` is the number of classes.
- Splits data into 80% training and 20% validation sets, stratified by label to maintain class distribution.
- Checks for data leakage between splits.
- Loads unlabeled data, initializing zero topic probabilities.
- Logs dataset sizes and class distributions.

#### **Model Initialization**:
- Uses `TinyBERT_General_4L_312D` (a lightweight BERT variant) for efficiency.
- Initializes the tokenizer and model with the number of labels set to the number of unique topics.
- Moves the model to the device (GPU/CPU).

#### **DataLoaders**:
- Creates `DataLoader` instances for training, validation, and unlabeled data.
- Uses `pin_memory=True` for faster data transfer to GPU and shuffles the training data.

#### **Optimizer and Scheduler**:
- Uses `AdamW` optimizer with the specified learning rate.
- Applies a linear learning rate scheduler with warmup (`get_linear_schedule_with_warmup`) to gradually increase the learning rate for stability.

#### **Training Loop**:
- Runs for up to 3 iterations, each with multiple epochs:
  - Trains the model using `train_model`, logging the loss.
  - Evaluates using `evaluate_classification_metrics`, logging accuracy, precision, recall, F1, and per-class F1 scores.
  - Saves the model after each iteration.
  - Stops early if the F1 score stabilizes (change < 0.01 and F1 ≥ 0.7).
  - If F1 < 0.7, generates pseudo-labels for unlabeled data, adds them to the training set, updates class weights, and creates a new `train_loader`.

#### **Final Steps**:
- Performs a final evaluation and saves the model to `saved_models/final_model`.
- Logs completion and final metrics.

---

### **Big Picture**
This script fine-tunes a TinyBERT model for text classification, using pseudo-labeled data from the LDA script and incorporating topic probabilities to guide training. It:
1. Loads and preprocesses pseudo-labeled and unlabeled reviews.
2. Trains TinyBERT with a custom topic-informed loss, balancing cross-entropy with KL divergence to align with LDA topics.
3. Iteratively improves the model by generating high-confidence pseudo-labels for unlabeled data.
4. Evaluates performance with detailed metrics and visualizes embeddings to assess class separation.
5. Saves models and logs for reproducibility.

The script builds on the LDA output, using its pseudo-labels as a starting point and enhancing them with BERT’s contextual understanding, making it ideal for classifying reviews into predefined categories with high accuracy.

---

### **Key Features and Why They Matter**
- **Topic-Informed Loss**: Integrates LDA topic probabilities, ensuring the model respects the thematic structure from the LDA script.
- **Pseudo-Labeling**: Expands the training set with high-confidence predictions, addressing data scarcity.
- **Class Weights**: Handles imbalanced classes, ensuring fair learning across topics.
- **UMAP Visualization**: Provides insight into how well the model separates classes in the embedding space.
- **Early Stopping**: Prevents overfitting by stopping when performance stabilizes.
- **Data Leakage Check**: Ensures robust evaluation by preventing train-validation overlap.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### **What is the Topic-Informed Loss?**
The `topic_informed_loss` function is a custom loss function that combines two types of loss to train the BERT model more effectively. It’s like a chef blending two ingredients to make a dish tastier: one ingredient ensures the model predicts the correct class labels (like “customer service” or “product quality”), while the other ensures the model’s predictions align with topic probabilities from the LDA model (from the previous script). This dual approach helps the BERT model learn from both labeled data and the thematic structure uncovered by LDA, making it more accurate and robust, especially when working with pseudo-labeled data.

Here’s the function from the script:

```python
def topic_informed_loss(logits, labels, topic_probs, class_weights, device, alpha=0.7):
    """Compute topic-informed loss combining cross-entropy and KL divergence."""
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))(logits, labels)
    soft_loss = torch.nn.KLDivLoss(reduction='batchmean')(torch.nn.functional.log_softmax(logits, dim=-1), topic_probs)
    return alpha * ce_loss + (1 - alpha) * soft_loss
```

Let’s unpack it step by step.

---

### **The Two Ingredients: Cross-Entropy and KL Divergence**
The topic-informed loss is a weighted combination of two losses:
1. **Cross-Entropy Loss**: This measures how well the model’s predictions match the true class labels (e.g., is this review about “customer service” or “product quality”?).
2. **KL Divergence Loss**: This measures how similar the model’s predicted probability distribution over classes is to the topic probability distribution from the LDA model.

The final loss is calculated as:
```
Loss = alpha * Cross-Entropy Loss + (1 - alpha) * KL Divergence Loss
```
where `alpha=0.7` means the cross-entropy loss contributes 70% to the total loss, and KL divergence contributes 30%. This balance ensures the model prioritizes correct label predictions but also respects the thematic insights from LDA.

---

### **Ingredient 1: Cross-Entropy Loss**
**What it does**: Cross-entropy loss is the standard loss function for classification tasks. It measures the difference between the model’s predicted probabilities (from `logits`) and the true labels. The goal is to make the model confident in the correct class while minimizing confidence in incorrect ones.

**How it works in the script**:
- The model outputs `logits`, which are raw scores for each class (e.g., [2.5, -1.2, 0.8] for three classes).
- `torch.nn.CrossEntropyLoss` applies softmax to convert logits into probabilities (e.g., [0.78, 0.04, 0.18]) and computes the loss by comparing these to the true labels (e.g., class 0 for “customer service”).
- The `class_weights` parameter adjusts the loss to handle imbalanced classes. For example, if “customer service” has fewer reviews than “product quality,” its weight is higher, so the model pays more attention to it.
- The weights are moved to the `device` (GPU/CPU) to ensure compatibility with the model’s computations.

**Why it’s important**: Cross-entropy ensures the model learns to classify reviews correctly based on the labeled data (from `selected_pseudo_labeled.csv`). It’s the primary driver of classification accuracy.

**Simple analogy**: Imagine you’re teaching a child to sort toys into boxes (e.g., cars vs. dolls). Cross-entropy is like checking if they put the car in the “car” box. If they get it wrong, you gently correct them, with more emphasis on rare toys (thanks to class weights).

---

### **Ingredient 2: KL Divergence Loss**
**What it does**: Kullback-Leibler (KL) divergence measures how different two probability distributions are. Here, it compares the model’s predicted class probabilities to the topic probabilities from the LDA model (stored in `topic_probs`). This encourages the model to make predictions that align with the thematic structure identified by LDA.

**How it works in the script**:
- The `logits` are converted to probabilities using `torch.nn.functional.log_softmax` (log-softmax is used for numerical stability in KL divergence).
- The `topic_probs` are arrays from the LDA script (e.g., [0.8, 0.15, 0.05] for a review, indicating 80% “customer service,” 15% “product quality,” 5% “pricing”).
- `torch.nn.KLDivLoss(reduction='batchmean')` computes the KL divergence, penalizing the model if its predicted probabilities (e.g., [0.78, 0.04, 0.18]) deviate from the LDA topic probabilities.
- The `reduction='batchmean'` averages the loss across the batch for stable training.

**Why it’s important**: The LDA model provides a thematic understanding of the reviews (e.g., grouping words like “support” and “response” into a “customer service” topic). KL divergence ensures the BERT model’s predictions respect these topics, especially for pseudo-labeled data where labels come from LDA. This is crucial when training on noisy or pseudo-labeled data, as it guides the model toward meaningful patterns.

**Simple analogy**: Think of LDA as a wise librarian who’s already grouped books by themes (e.g., “adventure” or “mystery”). KL divergence is like asking the child sorting toys to also consider the librarian’s groupings, ensuring their sorting (predictions) aligns with the broader themes.

---

### **Combining the Ingredients: The Weighted Sum**
The final loss is:
```
Loss = 0.7 * Cross-Entropy Loss + 0.3 * KL Divergence Loss
```
- **Alpha=0.7**: The script prioritizes cross-entropy (70%) to focus on correct label predictions, as this is the primary goal of classification.
- **1-alpha=0.3**: The 30% weight for KL divergence ensures the model incorporates LDA’s topic insights without overshadowing the labeled data.

This balance is like seasoning a dish: too much cross-entropy might ignore the LDA’s thematic guidance, while too much KL divergence might make the model overly reliant on potentially noisy topic probabilities. The choice of `alpha=0.7` is a reasonable starting point, emphasizing classification accuracy while leveraging topic information.

---

### **Why Use a Topic-Informed Loss?**
The topic-informed loss is a clever way to combine the strengths of two models:
- **BERT’s Strength**: BERT (TinyBERT in this case) excels at understanding contextual relationships in text, capturing nuances in review sentiment and meaning.
- **LDA’s Strength**: The LDA model (from the previous script) identifies thematic patterns across reviews, providing a probabilistic topic distribution (e.g., how much a review relates to “customer service” vs. other topics).

By combining these:
- **Improved Robustness**: The LDA topic probabilities guide BERT when labels are noisy or sparse (common with pseudo-labeled data).
- **Better Generalization**: Aligning with LDA topics helps the model capture broader themes, reducing overfitting to specific labeled examples.
- **Leveraging Unlabeled Data**: The pseudo-labeling process (in `generate_pseudo_labels`) uses LDA-derived topic probabilities to assign labels to unlabeled reviews, and the KL divergence ensures these are consistent with BERT’s predictions.

**Real-world analogy**: Imagine you’re a chef (BERT) trying to classify dishes as “Italian” or “Mexican.” You have some labeled examples (pseudo-labeled data), but also a food critic’s notes (LDA topics) saying a dish is 80% Italian based on ingredients. The topic-informed loss is like using both your taste tests (cross-entropy) and the critic’s notes (KL divergence) to decide, ensuring your classification respects both direct evidence and broader patterns.

---

### **How It Fits into the Script**
The `topic_informed_loss` is used in the `train_model` function during training:
- For each batch, the model processes input reviews (`input_ids`, `attention_mask`) to produce `logits`.
- The true labels (`labels`) and LDA topic probabilities (`topic_probs`) are passed to `topic_informed_loss`.
- The computed loss is used to update the model’s parameters via backpropagation (`loss.backward()`, `optimizer.step()`).

The script’s data comes from:
- **Pseudo-labeled data** (`selected_pseudo_labeled.csv`): Contains reviews with LDA-assigned topics, labels, and topic probabilities (e.g., [0.8, 0.15, 0.05]).
- **Unlabeled data** (`non_high_confidence_reviews.csv`): Used for pseudo-labeling, where the model assigns labels and generates one-hot topic probabilities (e.g., [1.0, 0.0, 0.0] for high-confidence predictions).

The topic-informed loss ensures that during training:
- The model learns to predict the correct labels (from pseudo-labeled data).
- Its predicted probabilities align with the LDA’s topic distributions, especially for reviews where LDA provided the initial labels.

---

Let’s dive into the part of the script where class weights are computed dynamically, explaining it in a simple, thorough, and engaging way, as if we’re unraveling a clever strategy to make our model fairer and more effective. I’ll pinpoint where this happens in the code, clarify what dynamic class weights are, why they’re used, and how they impact the training of the BERT model in the context of this script. This will tie directly into the topic-informed loss and the overall goal of classifying reviews accurately, especially when dealing with imbalanced data.

---

### **Where Class Weights Are Computed Dynamically**
The dynamic computation of class weights occurs in the `main` function, within the data loading section, and is updated again during the pseudo-labeling loop when new pseudo-labeled data is added. Here’s the relevant code snippet for the initial computation:

```python
# Compute class weights dynamically
N = len(pseudo_df)
C = len(topic_to_label)
class_counts = pseudo_df['label'].value_counts().to_dict()
class_weights = np.zeros(C)
for label in range(C):
    count = class_counts.get(label, 1)  # Avoid division by zero
    class_weights[label] = N / (count * C)
class_weights = torch.tensor(class_weights, dtype=torch.float)
```

And later, in the pseudo-labeling loop, when new pseudo-labeled data is added:

```python
# Update class weights after adding pseudo-labels
N = len(train_df)
class_counts = train_df['label'].value_counts().to_dict()
class_weights = np.zeros(C)
for label in range(C):
    count = class_counts.get(label, 1)  # Avoid division by zero
    class_weights[label] = N / (count * C)
class_weights = torch.tensor(class_weights, dtype=torch.float)
```

Let’s break this down step by step, exploring what’s happening and why it matters.

---

### **What Are Class Weights, and What Does “Dynamic” Mean?**
**Class weights** are numbers assigned to each class (e.g., “customer service,” “product quality”) to adjust the importance of that class during training. They’re used to address **class imbalance**, where some classes have many more examples than others. For example, if 80% of your reviews are about “product quality” and only 10% are about “customer service,” a model might overfocus on the majority class and perform poorly on the minority one. Class weights give more importance to underrepresented classes, balancing the model’s learning.

**Dynamic computation** means the weights are calculated based on the actual data distribution in the dataset at a given point, rather than using fixed or manually set values. In this script, the weights are computed initially from the pseudo-labeled dataset (`pseudo_df`) and updated whenever new pseudo-labeled data is added to the training set (`train_df`). This ensures the weights reflect the current class distribution, especially as the dataset grows with pseudo-labels.

---

### **Breaking Down the Code**
Let’s walk through the initial class weight computation:

1. **Dataset Size and Number of Classes**:
   - `N = len(pseudo_df)`: Gets the total number of reviews in the pseudo-labeled dataset.
   - `C = len(topic_to_label)`: Gets the number of unique classes (topics, like “customer service,” “product quality”), derived from unique values in `pseudo_df['topic']`.

2. **Count Class Frequencies**:
   - `class_counts = pseudo_df['label'].value_counts().to_dict()`: Creates a dictionary where keys are class labels (integers, e.g., 0 for “customer service”) and values are the number of reviews for that class. For example:
     ```python
     class_counts = {0: 500, 1: 200, 2: 50}  # 500 reviews for class 0, 200 for class 1, 50 for class 2
     ```

3. **Compute Weights**:
   - `class_weights = np.zeros(C)`: Initializes an array of zeros for the `C` classes.
   - For each class label (0 to C-1):
     - `count = class_counts.get(label, 1)`: Gets the number of reviews for the class, defaulting to 1 to avoid division by zero (in case a class has no examples).
     - This formula gives **higher weights to classes with fewer examples** (minority classes) and lower weights to classes with more examples (majority classes).

4. **Convert to Tensor**:
   - `class_weights = torch.tensor(class_weights, dtype=torch.float)`: Converts the weights to a PyTorch tensor for use in the loss function.

**Update in Pseudo-Labeling**:
When pseudo-labels are added (in `generate_pseudo_labels`), the training set (`train_df`) grows. The script recomputes `class_counts` and `class_weights` using the same formula to reflect the new distribution, ensuring weights stay relevant as the dataset evolves.

---

### **Why Compute Class Weights Dynamically?**
Dynamic class weights are crucial for several reasons:

1. **Handling Class Imbalance**:
   - In review datasets, some topics (e.g., “product quality”) may have many more reviews than others (e.g., “pricing”). Without weights, the model might prioritize the majority class, ignoring minority classes.
   - The formula \( \frac{N}{\text{count}_c \cdot C} \) gives higher weights to rare classes, so the model pays more attention to them during training. This improves performance on underrepresented topics.

2. **Adapting to Pseudo-Labeling**:
   - The script iteratively adds pseudo-labeled reviews to the training set. As new reviews are labeled, the class distribution may change (e.g., more reviews might be assigned to “customer service”).
   - Dynamic computation ensures weights reflect the current dataset, preventing outdated weights from skewing the model’s focus.

3. **Improving Model Fairness**:
   - By weighting classes inversely proportional to their frequency, the model learns to perform well across all classes, not just the dominant ones. This is critical for balanced metrics like F1 score, which the script tracks.

4. **Robustness in Semi-Supervised Learning**:
   - The pseudo-labeled data (from the LDA script) may have noisy or unevenly distributed labels. Dynamic weights adapt to these changes, ensuring the model doesn’t overfit to noisy majority classes.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

### **What is BERT, and What Does It Do in the Script?**
Here’s what the BERT model does in the script:
1. **Takes Reviews as Input**: It processes review text (from `selected_pseudo_labeled.csv` and `non_high_confidence_reviews.csv`) to understand their meaning.
2. **Predicts Topic Labels**: It assigns each review to a topic by outputting probabilities for each class (topic).
3. **Uses Topic-Informed Loss**: It’s trained with a custom loss that combines cross-entropy (for accurate label predictions) and KL divergence (to align with LDA topic probabilities).
4. **Handles Class Imbalance**: Uses dynamic class weights to focus on minority classes.
5. **Generates Pseudo-Labels**: Labels unlabeled reviews with high confidence to expand the training set.
6. **Visualizes Embeddings**: Produces embeddings for visualization to check how well topics are separated.

---

### **How BERT Works: The Big Picture**
BERT is like a super-smart librarian who reads a review and understands its context by considering every word’s relationship to all others in the sentence. Unlike older models that read text sequentially (left-to-right or right-to-left), BERT processes the entire sentence at once, capturing nuanced meanings. For example, in the review “The support was fast but the product was faulty,” BERT understands that “support” relates to positive customer service, while “product” relates to negative quality, thanks to its bidirectional context.

BERT consists of:
- **Input Processing**: Converts text into a format the model can understand (tokens and embeddings).
- **Transformer Layers**: Processes tokens to create contextual representations.
- **Classification Head**: Outputs probabilities for each topic.

Let’s break down how BERT works in the script, step by step.

---

### **Step 1: Input Processing with BERT Tokenizer**
**What Happens**:
- The script uses `BertTokenizer` to convert review text into a format BERT can process.
- In the `ReviewDataset` class, each review is tokenized:
  ```python
  encoding = self.tokenizer(
      text,
      max_length=self.max_length,  # 128 tokens
      padding='max_length',
      truncation=True,
      return_tensors='pt'
  )
  ```
- The tokenizer:
  - **Splits text into tokens**: Breaks the review into words or subwords (e.g., “supporting” might become “support” + “##ing”).
  - **Adds special tokens**: Inserts `[CLS]` at the start (for classification) and `[SEP]` at the end.
  - **Converts to IDs**: Maps tokens to numerical IDs from BERT’s vocabulary.
  - **Creates attention mask**: A binary mask (1 for real tokens, 0 for padding) to tell BERT which tokens to focus on.
  - **Pads/truncates**: Ensures all inputs are 128 tokens long, padding with zeros or cutting off excess.
- Output: A dictionary with `input_ids` (token IDs), `attention_mask`, `labels` (class ID), `topic_probs` (from LDA), and the original `text`.

**Example**:
For the review “Fast support, great service”:
- Tokens: `[CLS], fast, support, ,, great, service, [SEP]`
- Input IDs: `[101, 1234, 5678, 1010, 7890, 3456, 102]` (example IDs)
- Attention Mask: `[1, 1, 1, 1, 1, 1, 1]` (all tokens are valid)
- Label: `0` (e.g., “customer service”)
- Topic Probabilities: `[0.8, 0.15, 0.05]` (from LDA, indicating mostly “customer service”)

**Why It Matters**: This step transforms raw text into a numerical format that BERT can process, preserving the meaning and context of the review.

---

### **Step 2: BERT’s Transformer Layers**
**What Happens**:
- The `BertForSequenceClassification` model processes the tokenized input through its transformer layers.
- TinyBERT has 4 layers and 312 hidden units (smaller than standard BERT’s 12 layers and 768 units), making it faster and lighter while still effective.
- Each layer:
  - **Embeds Tokens**: Converts input IDs into high-dimensional vectors (embeddings) that combine word, position, and token-type information.
  - **Applies Attention**: Uses self-attention to weigh the importance of each token relative to others, capturing context (e.g., “support” is positive because it’s near “fast” and “great”).
  - **Transforms Embeddings**: Passes embeddings through feed-forward networks and normalization to refine representations.
- The final layer produces a hidden state for each token, with the `[CLS]` token’s embedding (a 312-dimensional vector in TinyBERT) summarizing the entire review.

**How It Works**:
- **Self-Attention**: For each token, BERT calculates how much it relates to every other token, creating contextual embeddings. For example, “support” is understood in the context of “fast” and “service.”
- **Bidirectional Context**: Unlike sequential models, BERT looks at the entire review at once, so it understands complex relationships (e.g., “not great” vs. “great”).
- **Output**: A matrix of hidden states (batch_size × sequence_length × 312), where the `[CLS]` embedding is used for classification.

**Why It Matters**: These layers allow BERT to capture the nuanced meaning of reviews, making it excellent for distinguishing between topics like “customer service” and “product quality” based on context.

---

### **Step 3: Classification Head**
**What Happens**:
- The `[CLS]` embedding from the final layer is passed to a classification head (a linear layer followed by softmax).
- The head outputs `logits`, raw scores for each class (e.g., `[2.5, -1.2, 0.8]` for 3 classes).
- In `evaluate_classification_metrics` and `generate_pseudo_labels`, logits are converted to probabilities using `torch.softmax` (e.g., `[0.78, 0.04, 0.18]`).
- The predicted class is the one with the highest probability (e.g., class 0 for “customer service”).

**Why It Matters**: The classification head translates BERT’s understanding of the review into a topic prediction, enabling the model to assign labels like those from the LDA script.

---

### **Step 4: Training with Topic-Informed Loss**
**What Happens**:
- The script trains the model using the `train_model` function, which calls `topic_informed_loss`:
  ```python
  loss = topic_informed_loss(outputs.logits, labels, topic_probs, class_weights, device)
  ```
- **Inputs to Loss**:
  - `logits`: Model’s raw scores for each class.
  - `labels`: True class IDs (e.g., 0 for “customer service”).
  - `topic_probs`: LDA topic probabilities (e.g., `[0.8, 0.15, 0.05]`).
  - `class_weights`: Dynamically computed weights (e.g., `[0.5, 1.25, 5.0]`) to handle class imbalance.
  - `device`: Ensures computations are on GPU/CPU.
- **Loss Calculation**:
  - **Cross-Entropy Loss**: Measures how well logits match true labels, weighted by `class_weights` to prioritize minority classes.
  - **KL Divergence Loss**: Encourages the model’s predicted probabilities to align with LDA topic probabilities.
  - Combined: `0.7 * ce_loss + 0.3 * soft_loss` (with `alpha=0.7`).
- **Backpropagation**:
  - The loss is used to compute gradients, which update the model’s weights via the `AdamW` optimizer.
  - A learning rate scheduler (`get_linear_schedule_with_warmup`) adjusts the learning rate, starting low for stability and decreasing over time.

**How Dynamic Class Weights Fit In**:
- The `class_weights` are applied in the cross-entropy loss:
  ```python
  ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))(logits, labels)
  ```
- Higher weights for minority classes (e.g., 5.0 for a rare class) make errors on those classes more costly, ensuring the model learns to classify them well.

**Why It Matters**: The topic-informed loss leverages both labeled data (pseudo-labels from LDA) and thematic insights (LDA topic probabilities), while class weights ensure balanced learning across imbalanced topics. This makes the model robust, especially for semi-supervised learning with pseudo-labeled data.

---

### **Step 5: Pseudo-Labeling and Iterative Training**
**What Happens**:
- The script runs up to 3 iterations, each with multiple epochs:
  - Trains the model on the current training set (`train_df`).
  - Evaluates on the validation set, logging metrics (accuracy, precision, recall, F1, per-class F1).
  - Saves the model after each iteration.
- If the F1 score is below 0.7, the `generate_pseudo_labels` function:
  - Uses the model to predict labels for unlabeled reviews (`non_high_confidence_reviews.csv`).
  - Selects predictions with confidence > 0.8, creating one-hot topic probabilities (e.g., `[1.0, 0.0, 0.0]`).
  - Adds these to `train_df`, updating class weights to reflect the new distribution.
- Stops early if the F1 score stabilizes (change < 0.01 and F1 ≥ 0.7).

**Why It Matters**: Pseudo-labeling expands the training set with high-confidence predictions, leveraging unlabeled data to improve the model. Dynamic class weights ensure the model adapts to the changing class distribution, maintaining fairness.

---

### **Step 6: Evaluation and Visualization**
**What Happens**:
- The `evaluate_classification_metrics` function:
  - Computes accuracy, precision, recall, and weighted F1 score.
  - Logs per-class F1 scores for each topic (e.g., “customer service: 0.85”).
  - Extracts `[CLS]` embeddings and uses UMAP to create 2D scatter plots, saved as PNGs (e.g., `embedding_iter1_epoch1.png`).
- Final metrics are logged, and the model is saved to `saved_models/final_model`.

**Why It Matters**: Metrics show how well the model classifies reviews, while UMAP plots visualize how well topics are separated in the embedding space, validating the model’s effectiveness.

------------------------------------------------------------------------------------------------------------------------------------------------------------------


Baik, saya akan menjelaskan lebih mendalam tentang **bagaimana TinyBERT bekerja** dalam konteks kode yang Anda berikan, dengan bahasa yang mudah dipahami. Saya akan fokus pada mekanisme inti TinyBERT, bagaimana ia memproses teks, menghasilkan prediksi, dan dilatih, sambil menjaga penjelasan tetap sederhana.

---

### **Apa itu TinyBERT?**
TinyBERT adalah versi "ringan" dari model BERT (Bidirectional Encoder Representations from Transformers). Bayangkan BERT seperti sebuah perpustakaan besar yang bisa memahami bahasa dengan sangat baik, tapi membutuhkan banyak tenaga komputer. TinyBERT adalah versi yang lebih kecil dan hemat energi, tetapi tetap pintar dalam memahami teks. Dalam kode ini, TinyBERT digunakan untuk mengklasifikasi ulasan teks (misalnya, menentukan apakah ulasan itu positif, negatif, atau terkait topik tertentu).

TinyBERT dibuat melalui proses yang disebut **distilasi pengetahuan**, di mana model besar (BERT) "mengajari" model kecil (TinyBERT) untuk meniru kemampuannya. Hasilnya, TinyBERT lebih cepat dan cocok untuk perangkat dengan sumber daya terbatas, seperti GPU biasa yang digunakan dalam kode.

---

### **Bagaimana TinyBERT Memproses Teks?**
Berikut adalah langkah-langkah cara TinyBERT bekerja dalam kode ini, dijelaskan dengan sederhana:

#### **1. Mengubah Teks Menjadi Angka (Tokenisasi)**:
- **Apa yang terjadi?** Teks ulasan (misalnya, "Restoran ini bagus sekali!") diubah menjadi format yang bisa dipahami oleh TinyBERT.
- **Caranya?** Kode menggunakan `BertTokenizer` untuk:
  - Memecah teks menjadi potongan kecil yang disebut **token** (biasanya kata atau bagian kata). Misalnya, "bagus sekali" bisa menjadi dua token: "bagus" dan "sekali".
  - Mengubah setiap token menjadi angka (disebut **input_ids**) berdasarkan kamus kata yang sudah dimiliki TinyBERT.
  - Menambahkan **attention_mask**, yaitu daftar yang memberi tahu model bagian mana dari teks yang penting (token asli) dan mana yang hanya pengisi (padding).
  - Memastikan semua teks memiliki panjang yang sama (maksimum 128 token) dengan memotong teks yang terlalu panjang atau menambahkan padding pada teks yang pendek.
- **Contoh**: Kalimat "Restoran ini bagus sekali!" diubah menjadi deretan angka seperti `[101, 1234, 5678, 9012, 3456, 102]` (101 dan 102 adalah kode khusus untuk awal dan akhir teks).

#### **2. Memahami Konteks Teks (Transformer)**:
- **Apa yang terjadi?** TinyBERT memproses deretan angka ini untuk memahami makna teks secara mendalam.
- **Caranya?** TinyBERT menggunakan arsitektur **Transformer**, yang merupakan inti dari cara kerjanya:
  - Transformer memiliki 4 lapisan (dalam TinyBERT versi ini), yang masing-masing berisi mekanisme **self-attention**. Self-attention memungkinkan model untuk "memperhatikan" semua kata dalam teks sekaligus dan menentukan hubungan antar kata.
  - Misalnya, dalam "Restoran ini bagus sekali!", self-attention membantu model memahami bahwa "bagus" terkait dengan "restoran" dan "sekali" memperkuat makna positifnya.
  - Setiap lapisan menghasilkan **representasi vektor** (seperti sidik jari digital) untuk setiap token, yang menangkap makna kata dalam konteks kalimat.
- **Hasilnya**: TinyBERT menghasilkan vektor khusus untuk token `[CLS]` (token awal), yang merangkum makna keseluruhan kalimat. Vektor ini seperti ringkasan dari seluruh teks.

#### **3. Membuat Prediksi (Klasifikasi)**:
- **Apa yang terjadi?** TinyBERT menggunakan vektor `[CLS]` untuk memprediksi kelas atau label teks (misalnya, "positif", "negatif", atau topik tertentu).
- **Caranya?**
  - Vektor `[CLS]` dikirim ke lapisan klasifikasi (seperti lapisan tambahan di atas Transformer) yang menghasilkan **logits** (skor untuk setiap kelas).
  - Logits ini diubah menjadi probabilitas menggunakan fungsi **softmax**. Misalnya, jika ada 3 kelas (positif, netral, negatif), TinyBERT mungkin menghasilkan probabilitas seperti `[0.8, 0.15, 0.05]`, yang berarti 80% kemungkinan positif.
  - Label dengan probabilitas tertinggi dipilih sebagai prediksi akhir.
- **Contoh**: Untuk ulasan "Restoran ini bagus sekali!", TinyBERT mungkin memprediksi label "positif" karena vektor `[CLS]` menangkap konteks positif dari teks.

#### **4. Pelatihan Model**:
- **Apa yang terjadi?** TinyBERT belajar dari data untuk membuat prediksi yang lebih akurat.
- **Caranya?**
  - Kode menggunakan **topic-informed loss**, yaitu kombinasi dua jenis kerugian:
    - **Cross-Entropy Loss**: Mengukur seberapa jauh prediksi model dari label sebenarnya (misalnya, jika model memprediksi "netral" tapi labelnya "positif", kerugian akan besar).
    - **KL Divergence Loss**: Membandingkan prediksi model dengan probabilitas topik (dari model lain seperti LDA) untuk memastikan model juga selaras dengan informasi topik.
    - Bobot kelas (`class_weights`) digunakan untuk menangani ketidakseimbangan data, misalnya, jika ada lebih banyak ulasan positif daripada negatif.
  - Model dilatih menggunakan optimizer **AdamW**, yang menyesuaikan parameter model untuk mengurangi kerugian.
  - **Learning rate scheduling** dengan **warmup** memastikan model belajar secara bertahap di awal pelatihan untuk menghindari perubahan yang terlalu drastis.
- **Proses**: Selama pelatihan, TinyBERT memperbarui parameter internalnya (bobot Transformer) berdasarkan data pelatihan untuk meningkatkan akurasi prediksi.

#### **5. Pseudo-Labeling (Belajar dari Data Tanpa Label)**:
- **Apa yang terjadi?** TinyBERT memprediksi label untuk data yang tidak memiliki label (data tanpa kategori) dan menggunakan prediksi yang sangat yakin untuk memperluas data pelatihan.
- **Caranya?**
  - Model memproses data tanpa label (dari `unlabeled_loader`) dan menghasilkan probabilitas untuk setiap kelas.
  - Jika probabilitas tertinggi melebihi ambang batas (misalnya, 0.8), prediksi tersebut dianggap cukup yakin dan ditambahkan ke dataset pelatihan sebagai **pseudo-label**.
  - Misalnya, untuk ulasan tanpa label "Pelayanan sangat ramah", TinyBERT mungkin memprediksi "positif" dengan probabilitas 0.9, sehingga ulasan ini ditambahkan ke data pelatihan dengan label "positif".
  - Proses ini membantu model belajar dari lebih banyak data, meskipun awalnya data tersebut tidak memiliki label.

#### **6. Evaluasi dan Visualisasi**:
- **Apa yang terjadi?** TinyBERT dievaluasi untuk melihat seberapa baik performanya, dan representasi teks divisualisasikan.
- **Caranya?**
  - **Evaluasi**: Model diuji pada data validasi untuk menghitung metrik seperti akurasi, presisi, *recall*, dan *F1-score*. Ini menunjukkan seberapa sering model memprediksi dengan benar.
  - **Visualisasi**: Representasi vektor `[CLS]` dari teks diubah menjadi ruang 2D menggunakan **UMAP** (algoritma untuk mereduksi dimensi). Ini menghasilkan peta yang menunjukkan bagaimana ulasan-ulasan dikelompokkan berdasarkan kelasnya (misalnya, ulasan positif berkumpul di satu area, negatif di area lain).
  - Visualisasi ini membantu memahami apakah TinyBERT mampu memisahkan kelas dengan baik.

#### **7. Penyimpanan Model**:
- Setelah pelatihan, TinyBERT dan tokenizernya disimpan ke disk untuk digunakan kembali. Ini memungkinkan model untuk diuji atau digunakan pada data baru tanpa perlu dilatih ulang.

---

### **Mekanisme Inti TinyBERT**
- **Self-Attention**: TinyBERT "memperhatikan" semua kata dalam teks sekaligus untuk memahami hubungan antar kata. Misalnya, dalam "Makanan enak tapi pelayanan lambat", self-attention membantu model memahami bahwa "enak" terkait dengan "makanan" dan "lambat" terkait dengan "pelayanan".
- **Bidirectional Context**: Tidak seperti model lain yang hanya membaca teks dari kiri ke kanan, TinyBERT memahami konteks dari kedua arah (kiri dan kanan), sehingga lebih akurat dalam menangkap makna.
- **Efisiensi**: Dengan hanya 4 lapisan dan 312 dimensi (dibandingkan 12 lapisan dan 768 dimensi di BERT), TinyBERT lebih cepat dan hemat memori, cocok untuk pelatihan di GPU biasa.
- **Distilasi Pengetahuan**: TinyBERT sudah dilatih sebelumnya untuk meniru kemampuan BERT, sehingga meskipun kecil, ia tetap kuat untuk tugas seperti klasifikasi teks.

---

### **Analogi Sederhana**
Bayangkan TinyBERT sebagai seorang pustakawan cerdas yang membaca ulasan restoran:
1. **Tokenisasi**: Pustakawan menerjemahkan ulasan menjadi kode-kode kecil (seperti buku dengan nomor katalog).
2. **Transformer**: Pustakawan memahami makna ulasan dengan melihat semua kata sekaligus, mencari tahu mana yang penting dan bagaimana kata-kata saling terkait.
3. **Klasifikasi**: Pustakawan memutuskan apakah ulasan itu positif, negatif, atau netral berdasarkan pemahamannya.
4. **Pelatihan**: Pustakawan belajar dari contoh ulasan yang sudah diberi label untuk menjadi lebih akurat, dan kadang-kadang menebak label untuk ulasan baru (pseudo-labeling).
5. **Evaluasi**: Pustakawan diuji untuk melihat seberapa sering tebakannya benar, dan hasilnya divisualisasikan untuk memahami pola ulasan.

---

### **Mengapa TinyBERT Efektif di Kode Ini?**
- **Ringan dan Cepat**: TinyBERT memungkinkan pelatihan dan prediksi cepat di GPU biasa, seperti yang terdeteksi dalam kode (`torch.cuda.get_device_name(0)`).
- **Pemahaman Konteks**: Mekanisme self-attention membuatnya sangat baik dalam memahami ulasan, bahkan jika teksnya kompleks.
- **Fleksibel dengan Pseudo-Labeling**: TinyBERT bisa memanfaatkan data tanpa label untuk meningkatkan pelatihan, yang sangat berguna ketika data berlabel terbatas.
- **Topic-Informed Loss**: Kombinasi kerugian memungkinkan TinyBERT belajar tidak hanya dari label, tetapi juga dari informasi topik, sehingga prediksinya lebih terarah.

---

### **Kesimpulan**
TinyBERT bekerja dengan mengubah teks menjadi angka (tokenisasi), memahami konteks teks menggunakan Transformer dengan self-attention, dan membuat prediksi kelas menggunakan vektor `[CLS]`. Dalam kode ini, TinyBERT dilatih dengan pendekatan cerdas yang menggabungkan data berlabel dan tanpa label (pseudo-labeling), serta kerugian khusus untuk memanfaatkan informasi topik. Visualisasi embedding membantu memahami bagaimana model memisahkan kelas. Dengan desainnya yang ringan, TinyBERT sangat efisien untuk tugas klasifikasi teks, bahkan pada perangkat dengan sumber daya terbatas.
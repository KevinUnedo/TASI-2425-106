# Identification of Functional and Non-Functional Requirements in MOOCs: Integrating Topic Modeling and Thematic Analysis

## Overview

This project aims to identify functional and non-functional requirements in Massive Open Online Courses (MOOCs) by integrating Topic Modeling and Thematic Analysis. The goal is to provide a more structured and comprehensive understanding of MOOCs user needs through the application of Seeded Latent Dirichlet Allocation (LDA) and Bidirectional Encoder Representations from Transformers (BERT). This approach uses a Systematic Literature Review (SLR) to identify relevant topics and categorize user reviews into Functional Requirements (FR) and Non-Functional Requirements (NFR).

## Abstract

Requirement Engineering (RE) is a critical phase in the software development cycle, where understanding user needs is paramount. In the context of MOOCs, user reviews serve as valuable data for identifying user preferences and requirements. However, the requirement elicitation process can be prone to ambiguities. This study proposes a systematic approach using LDA and BERT to categorize user feedback into FR and NFR, ultimately enhancing the efficiency of the requirement elicitation process.

## Project Structure

### 1. **Datasets Folder**
This folder contains various datasets used in the analysis and modeling.
- `Maalej_Dataset.csv`: Dataset used for user review analysis.
- `PROMISE_exp.csv`: Experimental dataset from the PROMISE repository.
- `PROMISE_exp_cleaned.csv`: Cleaned version of the PROMISE dataset.
- `preprocessed_coursera_review.csv`: Preprocessed user reviews from Coursera.
- `selected_pseudo_labeled_reviews.csv`: A subset of labeled reviews for training.
- `text_requirement.csv`: Dataset containing requirement text for analysis.

### 2. **Data Understanding Folder**
This folder contains Jupyter Notebooks for exploratory data analysis (EDA).
- `EDA_Coursera_reviews_dataset.ipynb`: Analysis of the Coursera reviews dataset.
- `EDA_PROMISE_Dataset_(Requirement_text).ipynb`: EDA on the PROMISE dataset focusing on requirement text.

### 3. **Data Preprocessing Folder**
This folder contains scripts for preprocessing and cleaning the datasets.
- `coursera_review.ipynb`: Script for processing Coursera reviews.
- `requirement_text_old.ipynb`: Older version of the requirement text processing script.
- `requirement_text.ipynb`: Final version of the requirement text preprocessing script.

### 4. **Model_LDA Folder**
This folder contains files related to the implementation of the LDA model for topic modeling.
- `lda_output.log`: Log file capturing the output of the LDA model.
- `seed_lda_labeling.py`: Python script for labeling topics with the LDA model.

### 5. **Model_TinyBERT Folder**
This folder contains files for training and using the TinyBERT model for requirement categorization.
- `train_model.py`: Python script to train the TinyBERT model.
- `training_TinyBERT.log`: Log file of the TinyBERT training process.
- `embeddings_plots`: Folder containing visualizations of the embeddings.
- `saved_models`: Folder containing saved models after training.

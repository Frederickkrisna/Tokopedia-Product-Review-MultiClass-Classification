# Tokopedia Product Review Multi-Class Classification

## Project Overview

This project implements a **multi-class text classification system** for Indonesian Tokopedia product reviews, going beyond traditional sentiment analysis to categorize reviews into **five actionable categories**. The system uses both **Machine Learning (SVM)** and **Deep Learning (IndoBERT)** approaches with special handling for imbalanced datasets.

### Categories

1. **Pujian** - Positive reviews praising products/services
2. **Kritik** - Negative reviews with complaints
3. **Saran** - Constructive feedback and improvement suggestions
4. **Informasi/Netral** - Factual statements without sentiment
5. **Lainnya** - Reviews that don't fit other categories

---

## Best Results

### Model Performance Comparison

| Model | Accuracy | F1-Macro | F1-Weighted |
|-------|----------|----------|-------------|
| SVM Baseline | 85% | 83% | 85% |
| SVM Tuned | 89% | 87% | 89% |
| **SVM Tuned Balanced** | 89% | **88%** | 89% |
| IndoBERT Baseline | 91% | 90% | 91% |
| IndoBERT Baseline Balanced | 91% | 90% | 91% |
| IndoBERT Tuned | 93% | 92% | 93% |
| **IndoBERT Tuned Balanced** | **93%** | **93%** | **93%** |

> **Note**: Balanced models show improved performance on minority classes (Saran, Lainnya)

---

## Project Structure

```
📦 Tokopedia-Product-Review-MultiClass-Classification
├── 📓 notebook.ipynb                    # Complete training pipeline
├── 🌐 app.py                            # Streamlit web application
├── 📋 requirements.txt                  # Python dependencies
├── 📊 df_clean.csv                      # Preprocessed dataset
├── 📄 tokopedia-product-reviews-2019.csv  # Original dataset (40K+ reviews)
│
├── 📂 models/                           # Trained models
│   ├── svm_baseline.pkl                 # SVM with default params
│   ├── svm_tuned.pkl                    # GridSearch optimized SVM
│   ├── svm_tuned_balanced.pkl           # SVM with class weights
│   ├── tfidf_vectorizer.pkl             # TF-IDF feature extractor
│   │
│   ├── indobert_p2_baseline/            # IndoBERT baseline model
│   ├── indobert_p2_baseline_balanced/   # IndoBERT baseline + class weights
│   ├── indobert_p2_tuned/               # Fine-tuned IndoBERT
│   └── indobert_p2_tuned_balanced/      # Fine-tuned IndoBERT + class weights
│
└── 📖 README.md                         # This file
```

---

## Installation & Setup

### Prerequisites

- **Python 3.8+** (tested on Python 3.10)
- **CUDA-compatible GPU** (optional, for faster IndoBERT inference)
- **8GB+ RAM** recommended
- **5GB+ disk space** for models

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd Tokopedia-Product-Review-MultiClass-Classification
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

**Option A: CPU Only (Faster Install)**
```bash
pip install -r requirements.txt
```

**Option B: With CUDA GPU Support (Recommended for faster inference)**
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install other requirements
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## How to Run

### Web Application (Streamlit)

Run the interactive web application for real-time classification:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

**Features:**
-  Real-time review classification
-  Select from 6 different models
-  Confidence scores visualization
-  Model performance comparison charts
-  Example reviews for testing

### Jupyter Notebook

To retrain models or explore the analysis:

```bash
jupyter notebook notebook.ipynb
```

**Notebook Contents:**
1. Data Loading & EDA
2. Text Preprocessing
3. Feature Engineering (TF-IDF)
4. Model Training (SVM + IndoBERT)
5. Evaluation & Comparison
6. Error Analysis
7. Visualization

---

## Technical Details

### Machine Learning Pipeline

**Preprocessing:**
- Lowercasing
- URL & mention removal
- Number removal
- Punctuation cleaning
- **Sastrawi stemming** (Indonesian)
- **Stopword removal** (Indonesian)

**Feature Extraction:**
- TF-IDF Vectorization
- Max features: 5,000
- N-grams: (1, 2)
- Min/max document frequency filtering

**Models:**

1. **SVM (Linear)**
   - `LinearSVC` from scikit-learn
   - C=1.0 (baseline), optimized via GridSearchCV (tuned)
   - `class_weight='balanced'` for handling imbalance

2. **IndoBERT (indobenchmark/indobert-base-p2)**
   - Pre-trained Indonesian BERT model
   - Fine-tuned on review dataset
   - Mixed precision training (FP16)
   - Custom loss with class weights for balanced variant
   - Batch size: 16, Learning rate: 2e-5 to 3e-5

### Class Weight Handling

For imbalanced datasets, we compute class weights:

```python
class_weights = compute_class_weight('balanced', 
                                    classes=np.unique(y_train), 
                                    y=y_train)
```

This ensures minority classes (Saran, Lainnya) get higher penalties during training.

---

## Dataset Information

- **Source**: Tokopedia Product Reviews (2019)
- **Size**: 40,365 reviews
- **Language**: Indonesian
- **Labels**: 5 categories
- **Split**: 70% train, 15% validation, 15% test
- **Class Distribution**: Imbalanced (Pujian is majority, Saran & Lainnya are minorities)

### Label Distribution

| Label | Count | Percentage |
|-------|-------|------------|
| Pujian | ~15,000 | 37% |
| Informasi/Netral | ~12,000 | 30% |
| Kritik | ~8,000 | 20% |
| Saran | ~3,000 | 7% |
| Lainnya | ~2,000 | 5% |

---

## Technical Stack

- **Language**: Python 3.12
- **NLP**: Sastrawi (Indonesian stemmer/stopwords)
- **ML**: scikit-learn (SVM, Logistic Regression, Naive Bayes)
- **Visualization**: matplotlib, seaborn, wordcloud
- **Data**: pandas, numpy

---

## Methodology

### 1. Data Collection & Cleaning
- 40,607 original reviews → 40,365 after cleaning
- Removed duplicates and missing values

### 2. Exploratory Data Analysis
- Text length distribution
- Rating analysis (highly imbalanced: 74.6% are 5-star)
- Word frequency analysis
- WordCloud visualization per rating

### 3. Label Engineering
- Rule-based labeling using keywords + ratings
- Validation through manual sampling
- Academic justification provided

### 4. Text Preprocessing (Indonesian-specific)
- Case folding
- Noise removal (URLs, special characters)
- Slang normalization (57 common Indonesian slang terms)
- Tokenization
- Stopword removal (Indonesian + custom)
- Stemming (Sastrawi)

### 5. Feature Extraction
- TF-IDF with 5,000 features
- Unigrams + bigrams
- 0.17% matrix density (sparse, efficient)

### 6. Modeling
- Trained 3 models: Naive Bayes, Logistic Regression, Linear SVM
- 70/15/15 train/val/test split with stratification
- Linear SVM achieved best performance

### 7. Evaluation
- Comprehensive metrics (accuracy, precision, recall, F1)
- Confusion matrix analysis
- Error analysis with examples
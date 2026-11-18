# Sarcasm Detection Using Traditional Machine Learning

## Project Overview

This project investigates the application of traditional supervised machine-learning techniques to sarcasm detection in news headlines.

The objective was to develop a Natural Language Processing (NLP) classification pipeline capable of classifying headlines into two categories:

- `0` – Non-sarcastic
- `1` – Sarcastic

The project compares Logistic Regression and Linear Support Vector Machine (SVM) classifiers using TF-IDF text representations. It also investigates dataset bias and the influence of publisher identity on classification performance.

## Technologies Used

- Python
- Google Colab
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Hugging Face Datasets
- Joblib
- Streamlit (interactive application)

## Dataset

The project uses the News Headlines Dataset for Sarcasm Detection by Rishabh Misra.

Original dataset:
- 26,709 headlines
- 3 columns: `article_link`, `headline`, `is_sarcastic`

The modelling dataset retained the headline text and binary sarcasm label. Duplicate headlines were removed, and conflicting labels were checked before training.

Final cleaned dataset:
- 26,602 unique headlines
- 14,951 non-sarcastic examples
- 11,651 sarcastic examples

The dataset was split into:
- Training set: 21,281 examples (80%)
- Testing set: 5,321 examples (20%)

Stratified splitting was used to preserve the class distribution.

## Methodology

### 1. Data preparation

- Selected headline text and sarcasm labels.
- Checked for missing values and duplicate headlines.
- Removed conflicting and duplicate records.
- Separated input features from target labels.

### 2. Text feature extraction

TF-IDF (Term Frequency–Inverse Document Frequency) was used to transform headline text into numerical feature vectors.

Configuration:
- Unigrams and bigrams (`ngram_range=(1, 2)`)
- Minimum document frequency of 2
- Maximum of 50,000 features
- Sublinear term-frequency scaling

### 3. Model development

Two supervised classification algorithms were implemented:

**Logistic Regression**

A linear classification algorithm used as the initial baseline model.

**Linear Support Vector Machine**

A linear-margin classifier evaluated as an alternative for text classification.

Both models were implemented using Scikit-learn pipelines combining TF-IDF vectorisation and classification.

### 4. Model evaluation

The models were assessed using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices
- Five-fold stratified cross-validation

## Experimental Results

### Held-out test set

| Metric | Logistic Regression | Linear SVM |
|---|---:|---:|
| Accuracy | 84.78% | 85.19% |
| Precision | 82.82% | 82.70% |
| Recall | 82.32% | 83.69% |
| F1-score | 82.57% | 83.19% |

### Five-fold cross-validation

| Metric | Logistic Regression | Linear SVM |
|---|---:|---:|
| Accuracy | 84.40% ± 0.59% | 84.74% ± 0.54% |
| Precision | 83.00% ± 0.65% | 82.95% ± 0.91% |
| Recall | 80.98% ± 0.76% | 82.03% ± 0.32% |
| F1-score | 81.98% ± 0.70% | 82.48% ± 0.55% |

Linear SVM achieved the strongest accuracy, recall and F1-score in these experiments and was selected as the preferred model.

## Dataset Bias Investigation

An additional experiment examined whether publisher identity was associated with sarcasm labels.

The publisher distribution revealed:

| Publisher | Non-sarcastic | Sarcastic |
|---|---:|---:|
| Huffington Post | 14,985 | 0 |
| The Onion | 0 | 11,724 |

A publisher-only Logistic Regression classifier achieved 100% accuracy on a random held-out split. A majority-class baseline achieved 56.10%.

This revealed perfect publisher-label confounding in the dataset. Publisher identity alone could predict the target label without analysing the headline.

Consequently, the headline model's performance cannot establish that it learned sarcasm independently of publisher-specific language patterns.

This is a major dataset limitation and an important finding of the investigation.

## Limitations

1. Publisher identity is perfectly associated with the target labels.
2. The model may learn source-related vocabulary and writing patterns.
3. Sarcasm can depend on context, tone, cultural knowledge and implied meaning that TF-IDF representations cannot fully capture.
4. Random train/test splitting does not establish generalisation to unseen publishers.
5. The current results should not be interpreted as proof of general-purpose sarcasm understanding.

## Future Work

- Investigate datasets containing both classes across multiple publishers.
- Evaluate models using source-aware and publisher-held-out experimental designs where class coverage permits.
- Compare traditional machine-learning approaches with contextual language models.
- Investigate model explainability and influential linguistic features.
- Explore applications of NLP classification techniques to cybersecurity text analytics.

## Project Status

The baseline model development, evaluation and publisher-bias investigation have been completed. A Streamlit application is planned to demonstrate headline classification using the saved Linear SVM pipeline.

## Research Focus

The project explores supervised NLP classification, model evaluation, dataset bias, source-related confounding and generalisation. These topics provide a foundation for further investigation into reliable machine-learning systems.

## Run it on your Windows computer

Open PowerShell inside the `sarcasm-detection` directory:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (normally http://localhost:8501).

The saved pipelines were created with scikit-learn 1.6.1. Newer scikit-learn versions can print `InconsistentVersionWarning` when loading them; predictions still work, but for exact reproducibility retrain and save the models with the same scikit-learn version used at runtime. The notebook in `notebooks/sarcasm_detection.ipynb` contains the full training and evaluation workflow.

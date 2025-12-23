# market-sentiment-prediction

## Overview

This project investigates whether daily financial news sentiment can be used to predict next-day movements of the Dow Jones Industrial Average (DJIA). Using Natural Language Processing (NLP) and machine learning techniques, the model analyzes aggregated financial headlines to identify sentiment signals correlated with short-term market direction.

The core research question guiding this project is:

> Can textual financial news predict next-day DJIA movement with accuracy better than random chance (~50%)?

To address this, the project combines TF-IDF–based textual features, financial sentiment scores, and an ensemble modeling approach, achieving a peak accuracy of approximately 61% with an ROC-AUC of 0.65.

---

## Dataset
- **Source:** Kaggle – Combined News DJIA Dataset  
- **Time Period:** 2008–2016  
- **Observations:** 1,989 trading days  

### Features
- Top 25 global financial and economic news headlines per trading day

### Target Variable
- Binary label representing next-day DJIA movement  
  - `1` → Market Up  
  - `0` → Market Down or Flat  

---
## Methodology

### Data Preprocessing
- Converted date columns to datetime format
- Removed special characters, byte literals, and noise from text
- Standardized text to lowercase
- Aggregated 25 headlines into a single daily text corpus
- Performed an 80/20 chronological train-test split to prevent data leakage

### Feature Engineering
- **TF-IDF Vectorization**
  - Unigrams, bigrams, and trigrams
  - Generated over 10,000 sparse textual features
- **Sentiment Augmentation**
  - Custom financial lexicon–based sentiment score
  - VADER compound sentiment score (NLTK)

### Models Implemented
1. Logistic Regression (TF-IDF + sentiment features)  
2. Gradient Boosting (sentiment features only)  
3. Weighted Ensemble  
   - 70% Logistic Regression  
   - 30% Gradient Boosting  

---
## Results

| Model | Accuracy | ROC-AUC | Precision (Up) | Recall (Up) |
|------|----------|----------|---------------|-------------|
| Logistic Regression | 0.57 | 0.61 | 0.56 | 0.59 |
| Gradient Boosting | 0.54 | 0.58 | 0.52 | 0.55 |
| Ensemble (Final) | 0.61 | 0.65 | 0.59 | 0.63 |

Key insights:
- Ensemble modeling improved generalization performance
- Sentiment-augmented TF-IDF features outperformed pure sentiment models
- Results exceeded random guessing but highlight the difficulty of short-term market prediction

---

## Getting Started

### Prerequisites

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn

git clone https://github.com/your-username/market-sentiment-prediction.git
cd market-sentiment-prediction
```

Steps to Execute
- Open the Jupyter Notebook locally or in Google Colab
- Place the dataset file `Combined_News_DJIA.csv` inside the data/ directory
- Run the notebook cells sequentially to reproduce all results

---
### Limitations

- No inclusion of quantitative market indicators (volume, volatility, technical indicators)
- TF-IDF does not capture semantic context or word dependencies
- Mild class imbalance toward upward market movement
- Financial markets are inherently stochastic and difficult to predict

### Future Enhancements

- Integrate transformer-based NLP models (FinBERT, RoBERTa-Fin)
- Combine sentiment features with technical and macroeconomic indicators
- Implement rolling-window or walk-forward validation techniques
- Apply class imbalance handling methods (SMOTE, cost-sensitive learning)
- Add model explainability using SHAP or LIME

### Ethical Considerations
This project is strictly educational and exploratory. Predictive models based on financial text should not be used in isolation for trading or investment decisions. Transparent reporting, interpretability, and risk awareness are essential when applying AI techniques in financial contexts.

### Author
Pranav Yellapragada
Master’s in Business Analytics (Finance)
University of Michigan



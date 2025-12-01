Fake News Classification — Machine Learning vs Large Language Models

**Author:** Ana Stevanović  
**Year:** 2025  
**Tech:** Python, Scikit-learn, Logistic Regression, TF-IDF, GPT-4o-mini, GPT-4.1-mini, GPT-4o  
**Goal:** Compare a classical ML pipeline with modern LLM zero-shot classifiers.

---

## Project Overview

This project evaluates two fundamentally different approaches to classifying news articles as **real (0)** or **fake (1)**.

### **1) Machine Learning Baseline**

A classical supervised ML pipeline:

- TF-IDF vectorizer
- Logistic Regression classifier
- Trained directly on the dataset
- Achieves **96.25% accuracy**

### **2) Large Language Models (LLMs)**

Zero-shot classification using:

- GPT-4o-mini
- GPT-4o-mini (enhanced prompting)
- GPT-4.1-mini
- GPT-4o

LLMs were **not trained** on the dataset — they rely solely on general world knowledge.  
Accuracy ranges from **26% to 65%**, depending on model size and prompting strategy.

---

## 🗂 Project Structure

project/
│── notebooks/
│ └── fake_real_news_ml_vs_llm.ipynb
│
│── data/
│ └── data.csv (not included in the repo)
│
│── results/
│── README.md
│── requirements.txt
│── .gitignore

yaml
Kopier kode

---

## Results Summary

| Model                 | Type          | Accuracy   |
| --------------------- | ------------- | ---------- |
| Logistic Regression   | Supervised ML | **0.9625** |
| GPT-4o-mini           | LLM zero-shot | 0.6500     |
| GPT-4o-mini (prompt+) | LLM zero-shot | 0.6500     |
| GPT-4.1-mini          | LLM zero-shot | 0.2625     |
| GPT-4o                | LLM zero-shot | 0.3375     |

### Key Finding

> Classical ML **wins easily** when trained on a domain-specific dataset.  
> LLMs **struggle** when forced to classify fake news without fine-tuning.

ML learns directly from the dataset.  
LLMs rely on general reasoning, which is not enough for nuanced fake-news detection.

---

## Methodology

### **1. Data Preparation**

- Combined headline and body into a single `text` field
- Removed null or malformed entries
- Basic EDA: distribution, text length analysis, histogram visualizations

### **2. Machine Learning Pipeline**

```python
Pipeline([
    ("tfidf", TfidfVectorizer(max_features=50000, stop_words="english")),
    ("lr", LogisticRegression(max_iter=200))
])
3. LLM Evaluation (Zero-Shot)
GPT models were queried with a classification prompt:

python
Kopier kode
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)
A sample of 80 test examples was evaluated for fair comparison.

 What I Learned

Supervised ML shines on structured, domain-specific tasks

LLM prompting alone cannot replace dataset-specific training

Zero-shot LLM predictions vary drastically by model size

Model bias and hallucinations must be measured, not assumed

ML interpretability (TF-IDF coefficients) is extremely valuable

 How to Run

Install dependencies
bash
Kopier kode
pip install -r requirements.txt
Set API key
bash
Kopier kode
export OPENAI_API_KEY="your_key_here"
Windows:

bash
Kopier kode
setx OPENAI_API_KEY "your_key_here"
Run the notebook
Open in Jupyter Lab or VS Code:

bash
Kopier kode
notebooks/fake_real_news_ml_vs_llm.ipynb

 Contact

For conversation, collaboration or technical discussion:
LinkedIn: https://www.linkedin.com/in/ana-stevanovic
```

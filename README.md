
# Twitter Sentiment Analysis using BERT

An end-to-end NLP project that performs **binary sentiment analysis (positive / negative)** on Twitter data using **DistilBERT**.  
The project follows a **production-style ML pipeline** with modular code, logging, exception handling, early stopping, and model persistence.

---

##  Project Overview

- Built a complete sentiment analysis pipeline using **transformer-based NLP**
- Fine-tuned **DistilBERT** on a balanced subset of Twitter data
- Implemented **early stopping** to prevent overfitting
- Logged training progress and saved the **best-performing model**
- Designed with a clean folder structure and reusable components

---

##  Model & Performance

- **Model:** DistilBERT (`distilbert-base-uncased`)
- **Task:** Binary sentiment classification
- **Dataset Size:** 160,000 tweets (balanced)
- **Evaluation Metrics:** Accuracy, F1-score
- **Best Accuracy:** ~82–85%
- **Early Stopping:** Enabled

---

##  Project Structure

```

sentiment_analysis_bert/
│
├── data/
│   └── tweets.csv
│
├── artifacts/
│   ├── model/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── tokenizer/
│   └── logs/
│       └── training_*.log
│
├── src/
│   ├── **init**.py
│   ├── logger.py
│   ├── exception.py
│   ├── data_ingestion.py
│   ├── dataset.py
│   ├── trainer.py
│   └── inference.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore

````

---

## ⚙️ Installation & Setup

### 1️ Create and activate virtual environment
```bash
python -m venv projectenv
projectenv\Scripts\activate
````

### 2️ Install dependencies

```bash
pip install -r requirements.txt
```

---

##  How to Train the Model

1. Place the dataset in:

```
data/tweets.csv
```

2. Run training:

```bash
python main.py
```

### What happens:

* Dataset is loaded and balanced
* DistilBERT is fine-tuned with early stopping
* Best model is saved to `artifacts/model/`
* Training logs are stored in `artifacts/logs/`

---

##  Inference (Test the Trained Model)

Run:

```bash
python src/inference.py
```

Example output:

```
Positive
Negative
```

---

##  Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Scikit-learn
* Pandas, NumPy
* NVIDIA CUDA (GPU acceleration)

---

##  Key Learnings

* Transformer-based NLP modeling
* Handling large-scale text data efficiently
* Early stopping to prevent overfitting
* Modular ML project design
* GPU-accelerated training with PyTorch

---

##  Future Improvements

* Train on larger dataset (80k–150k samples)
* Upgrade to RoBERTa-base
* Add Streamlit web application
* Deploy as REST API

---

##  Author
**Tatapudi Bhaskar Phaneendra**


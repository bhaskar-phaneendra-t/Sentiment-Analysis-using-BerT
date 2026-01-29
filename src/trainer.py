from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.dataset import TwitterDataset
from src.logger import get_logger

logger = get_logger()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

def train_model(df):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"],
        df["sentiment"],
        test_size=0.2,
        stratify=df["sentiment"],
        random_state=42
    )

    train_enc = tokenizer(
        train_texts.tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )

    test_enc = tokenizer(
        test_texts.tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )

    train_dataset = TwitterDataset(train_enc, train_labels)
    test_dataset = TwitterDataset(test_enc, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="artifacts/model",
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=200,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    logger.info("Starting training")
    trainer.train()

    logger.info("Saving best model and tokenizer")
    trainer.save_model("artifacts/model")
    tokenizer.save_pretrained("artifacts/model/tokenizer")

    return trainer

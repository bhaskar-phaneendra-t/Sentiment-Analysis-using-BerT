import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "artifacts/model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH + "/tokenizer")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return "Positive" if pred == 1 else "Negative"


if __name__ == "__main__":
    print(predict_sentiment("I absolutely loved this movie"))
    print(predict_sentiment("This is the worst experience ever"))

import numpy as np
from sklearn.metrics import classification_report

def evaluate_model(trainer, test_dataset):
    preds = trainer.predict(test_dataset)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)

    print(classification_report(y_true, y_pred))

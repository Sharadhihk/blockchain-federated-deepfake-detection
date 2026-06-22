import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from tensorflow.keras.models import load_model

from face_dataset_generator import FaceGenerator


# ==========================
# LOAD TEST DATA
# ==========================

test_gen = FaceGenerator(
    "face_test.csv",
    batch_size=1,
    shuffle=False
)

# ==========================
# LOAD MODEL
# ==========================

model = load_model(
    "face_models/final_federated_model.h5"
)

# ==========================
# PREDICTIONS
# ==========================

y_true = []
y_pred = []
y_prob = []

for X_batch, y_batch in test_gen:

    preds = model.predict(
        X_batch,
        verbose=0
    )

    y_true.extend(
        y_batch.astype(int)
    )

    y_prob.extend(
        preds.flatten()
    )

    y_pred.extend(
        (preds.flatten() > 0.5).astype(int)
    )

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# ==========================
# METRICS
# ==========================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

roc_auc = roc_auc_score(
    y_true,
    y_prob
)

cm = confusion_matrix(
    y_true,
    y_pred
)

# ==========================
# RESULTS
# ==========================

print("\n==============================")
print("FEDERATED MODEL RESULTS")
print("==============================")

print(f"\nAccuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")
print(f"ROC-AUC   : {roc_auc*100:.2f}%")

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Real", "Fake"]
    )
)
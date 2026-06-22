import tensorflow as tf
from tensorflow.keras.models import load_model

from face_dataset_generator import FaceGenerator

# ==========================
# LOAD TEST DATA
# ==========================

test_gen = FaceGenerator(
    r"D:\deepfake_training_gpu\face_test.csv",
    batch_size=2,
    shuffle=False
)

# ==========================
# LOAD FEDERATED MODEL
# ==========================

model = load_model(
    "face_models/final_federated_model.h5"
)

# ==========================
# EVALUATE
# ==========================

loss, accuracy = model.evaluate(
    test_gen,
    verbose=1
)

print("\n========================")
print("FEDERATED MODEL RESULTS")
print("========================")
print(f"Loss     : {loss:.4f}")
print(f"Accuracy : {accuracy*100:.2f}%")
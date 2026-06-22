import os

import tensorflow as tf
from tensorflow.keras.models import load_model

from face_dataset_generator import FaceGenerator
from fedavg import federated_average


# =====================================
# CONFIGURATION
# =====================================

ROUNDS = 3
LOCAL_EPOCHS = 1

GLOBAL_MODEL_PATH = r"face_models\best_finetuned_model_93_79.h5"

CLIENT_CSVS = [
    "face_train_client1.csv",
    "face_train_client2.csv",
    "face_train_client3.csv"
]

TEST_CSV = "face_test.csv"

SAVE_DIR = "face_models"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# =====================================
# GPU CHECK
# =====================================

gpus = tf.config.list_physical_devices('GPU')

print("\nDetected GPUs:")
print(gpus)

if len(gpus) > 0:

    try:

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )

        print("\nGPU Enabled!")

    except Exception as e:

        print("\nGPU Config Error:")
        print(e)

else:

    print("\nWARNING: No GPU Detected")


# =====================================
# LOAD TEST DATA
# =====================================

print("\nLoading Test Dataset...")

test_gen = FaceGenerator(
    TEST_CSV,
    batch_size=2,
    shuffle=False
)


# =====================================
# LOAD GLOBAL MODEL
# =====================================

print("\nLoading Global Model...")

global_model = load_model(
    GLOBAL_MODEL_PATH
)

print("\nGlobal Model Loaded Successfully")


# =====================================
# FEDERATED TRAINING
# =====================================

for round_num in range(ROUNDS):

    print(
        f"\n{'='*50}"
    )

    print(
        f"FEDERATED ROUND {round_num + 1}"
    )

    print(
        f"{'='*50}\n"
    )

    global_weights = global_model.get_weights()

    client_weights = []

    # ===============================
    # TRAIN EACH CLIENT
    # ===============================

    for client_id, csv_file in enumerate(
        CLIENT_CSVS,
        start=1
    ):

        print(
            f"\nTraining Client {client_id}"
        )

        train_gen = FaceGenerator(
            csv_file,
            batch_size=2,
            shuffle=True
        )

        client_model = load_model(
            GLOBAL_MODEL_PATH
        )

        client_model.set_weights(
            global_weights
        )

        client_model.fit(
            train_gen,
            epochs=LOCAL_EPOCHS,
            verbose=1
        )

        client_weights.append(
            client_model.get_weights()
        )

    # ===============================
    # FEDAVG
    # ===============================

    print("\nPerforming FedAvg...")

    averaged_weights = federated_average(
        client_weights
    )

    global_model.set_weights(
        averaged_weights
    )

    # ===============================
    # EVALUATION
    # ===============================

    print(
        "\nEvaluating Global Model..."
    )

    loss, accuracy = global_model.evaluate(
        test_gen,
        verbose=1
    )

    print(
        f"\nRound {round_num + 1}"
    )

    print(
        f"Loss     : {loss:.4f}"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    # ===============================
    # SAVE ROUND MODEL
    # ===============================

    round_model_path = os.path.join(
        SAVE_DIR,
        f"global_round_{round_num + 1}.h5"
    )

    global_model.save(
        round_model_path
    )

    print(
        f"\nSaved: {round_model_path}"
    )


# =====================================
# SAVE FINAL MODEL
# =====================================

FINAL_MODEL_PATH = os.path.join(
    SAVE_DIR,
    "final_federated_model.h5"
)

global_model.save(
    FINAL_MODEL_PATH
)

print("\n")
print("="*60)
print("FEDERATED TRAINING COMPLETE")
print("="*60)

print(
    f"\nFinal Model Saved At:\n{FINAL_MODEL_PATH}"
)
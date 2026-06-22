import cv2
import hashlib
import numpy as np

from retinaface import RetinaFace
from tensorflow.keras.models import load_model

from blockchain.blockchain_interface import store_record

IMG_SIZE = 224
SEQUENCE_LENGTH = 20

VIDEO_PATH = r"C:\Users\hrvis\Downloads\WhatsApp Video 2026-06-16 at 7.07.59 PM (1).mp4"

MODEL_PATH = r"face_models\final_federated_model.h5"


def get_video_hash(video_path):

    sha256 = hashlib.sha256()

    with open(video_path, "rb") as f:

        while True:

            chunk = f.read(4096)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def extract_face(frame):

    try:

        detections = RetinaFace.detect_faces(frame)

        if not isinstance(detections, dict):
            return None

        largest_area = 0
        best_face = None

        for face_id in detections:

            facial_area = detections[face_id]["facial_area"]

            x1, y1, x2, y2 = facial_area

            area = (x2 - x1) * (y2 - y1)

            if area > largest_area:

                largest_area = area
                best_face = facial_area

        if best_face is None:
            return None

        x1, y1, x2, y2 = best_face

        x1 = max(0, x1)
        y1 = max(0, y1)

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            return None

        face = cv2.resize(
            face,
            (IMG_SIZE, IMG_SIZE)
        )

        return face

    except Exception:

        return None


def extract_sequence(video_path):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:

        cap.release()

        raise Exception(
            "Could not read video."
        )

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH,
        dtype=int
    )

    frames = []

    for idx in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            idx
        )

        success, frame = cap.read()

        if not success:
            continue

        face = extract_face(frame)

        if face is None:
            continue

        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        face = face.astype(
            np.float32
        ) / 255.0

        frames.append(face)

    cap.release()

    if len(frames) == 0:

        raise Exception(
            "No faces detected."
        )

    while len(frames) < SEQUENCE_LENGTH:

        frames.append(
            frames[-1]
        )

    return np.array(
        frames,
        dtype=np.float32
    )


print("=" * 60)
print("DEEPFAKE DETECTION + BLOCKCHAIN VERIFICATION")
print("=" * 60)

print("\nLoading model...")

model = load_model(
    MODEL_PATH
)

print("Extracting faces from video...")

sequence = extract_sequence(
    VIDEO_PATH
)

print("Sequence Shape:", sequence.shape)

sequence = np.expand_dims(
    sequence,
    axis=0
)

prediction = model.predict(
    sequence,
    verbose=0
)[0][0]

print("\nRaw Prediction:", prediction)

if prediction > 0.5:

    prediction_label = "FAKE"

    confidence = prediction * 100

else:

    prediction_label = "REAL"

    confidence = (1 - prediction) * 100

print("\nPrediction:", prediction_label)
print(f"Confidence: {confidence:.2f}%")

video_hash = get_video_hash(
    VIDEO_PATH
)

print("\nVideo Hash:")
print(video_hash)

print("\nStoring prediction on Ethereum...")

tx_hash = store_record(
    video_hash,
    prediction_label
)

print("\nTransaction Hash:")
print(tx_hash)

print("\nProcess Completed Successfully.")
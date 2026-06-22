import os
import cv2
import numpy as np
import pandas as pd

from tensorflow.keras.utils import Sequence

IMG_SIZE = 224
SEQUENCE_LENGTH = 20


class FaceGenerator(Sequence):

    def __init__(
        self,
        csv_file,
        batch_size=4,
        shuffle=True
    ):

        self.df = pd.read_csv(csv_file)

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indexes = np.arange(
            len(self.df)
        )

        self.on_epoch_end()

    def __len__(self):

        return int(
            np.ceil(
                len(self.df) /
                self.batch_size
            )
        )

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indexes
            )

    def load_sequence(
        self,
        folder_path
    ):

        image_files = sorted([

            f for f in os.listdir(folder_path)

            if f.endswith(".jpg")

        ])

        frames = []

        for image_name in image_files[:SEQUENCE_LENGTH]:

            image_path = os.path.join(
                folder_path,
                image_name
            )

            img = cv2.imread(
                image_path
            )

            if img is None:
                continue

            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            # =====================
            # DATA AUGMENTATION
            # =====================

            if self.shuffle:

                # Horizontal Flip
                if np.random.rand() < 0.5:

                    img = cv2.flip(
                        img,
                        1
                    )

                # Brightness / Contrast
                if np.random.rand() < 0.3:

                    alpha = np.random.uniform(
                        0.9,
                        1.1
                    )

                    beta = np.random.uniform(
                        -10,
                        10
                    )

                    img = cv2.convertScaleAbs(
                        img,
                        alpha=alpha,
                        beta=beta
                    )

            img = img.astype(
                np.float16
            ) / 255.0

            frames.append(
                img
            )

        if len(frames) == 0:
            return None

        if len(frames) < SEQUENCE_LENGTH:

            while len(frames) < SEQUENCE_LENGTH:

                frames.append(
                    frames[-1]
                )

        return np.array(
            frames,
            dtype=np.float16
        )

    def __getitem__(
        self,
        index
    ):

        batch_indexes = self.indexes[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        X = []
        y = []

        for idx in batch_indexes:

            row = self.df.iloc[idx]

            sequence = self.load_sequence(
                row["folder_path"]
            )

            if sequence is None:
                continue

            X.append(sequence)

            y.append(
                row["label"]
            )

        return (
            np.array(
                X,
                dtype=np.float16
            ),
            np.array(
                y,
                dtype=np.float32
            )
        )
# Deepfake Detection using Federated Learning and Blockchain

## Overview

This project is a Deepfake Video Detection System that combines Deep Learning, Federated Learning, and Ethereum Blockchain.

The system uses RetinaFace for face detection, Xception CNN for feature extraction, and LSTM for learning temporal patterns from video frames. A simulated Federated Learning setup with 3 clients and FedAvg aggregation is used for collaborative model training. Prediction results are securely stored on the Ethereum Sepolia Blockchain for verification.

---

## Features

* Deepfake Video Detection
* RetinaFace Face Detection
* Xception CNN Feature Extraction
* LSTM Temporal Learning
* Federated Learning (FedAvg)
* 3 Virtual Clients
* Ethereum Blockchain Integration
* SHA256 Video Hashing
* Smart Contract Based Verification
* Tamper-Proof Prediction Storage

---

## System Workflow

Input Video

↓

Face Detection (RetinaFace)

↓

Frame Preprocessing

↓

Xception CNN

↓

LSTM

↓

Deepfake Classification

↓

Federated Learning (FedAvg)

↓

SHA256 Hash Generation

↓

Ethereum Blockchain

↓

Verification

---

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* RetinaFace
* Xception CNN
* LSTM
* Ethereum Sepolia
* Solidity
* Web3.py
* MetaMask
* Alchemy

---

## Dataset

Datasets used:

* FaceForensics++
* CelebDF

Dataset Distribution:

* Real Videos: 2253
* Fake Videos: 2254
* Total Videos: 4507

---

## Federated Learning Setup

* 3 Virtual Clients
* FedAvg Aggregation
* 3 Communication Rounds
* 1 Local Epoch per Round

This simulates collaborative learning while keeping the project lightweight and suitable for student-level hardware.

---

## Results

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 94.24% |
| Precision | 96.72% |
| Recall    | 91.57% |
| F1 Score  | 94.08% |
| ROC-AUC   | 98.39% |

---

## Blockchain Storage

The following information is stored on Ethereum:

* Video Hash
* Prediction Result
* Timestamp
* Transaction Record

This provides an immutable and verifiable record of every prediction.

---

## Future Improvements

* Real-Time Detection
* IPFS Integration
* Web Application Deployment
* Mobile Application Support
* Multi-Blockchain Support
* Explainable AI Features

---

## Authors

Developed as a B.Tech Capstone Project on Deepfake Detection using Federated Learning and Blockchain.

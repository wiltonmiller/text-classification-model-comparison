import pandas as pd
import numpy as np
import json
import pickle
import re

from data_cleaner import *

def predict_all(filename):
    """Takes raw CSV file, returns predicted labels."""
    df_raw = pd.read_csv(filename)
    df_clean = clean_dataframe(df_raw)
    X = extract_features(df_clean)

    W, B, class_ids = load_model_params("models/best_model_params.json")
    class_names = load_label_classes("models/label_classes.json")

    logits = X.dot(W.T) + B
    probs = softmax(logits)
    pred_ids = np.argmax(probs, axis=1)

    # Map numeric → label name
    id_to_label = {cid: cname for cid, cname in zip(class_ids, class_names)}
    preds = [id_to_label[i] for i in pred_ids]

    return preds

def accuracy_on_file(filename):
    df = pd.read_csv(filename)
    if "label" not in df.columns:
        print("No labels found in this file.")
        return None

    true_labels = df["label"].tolist()
    preds = predict_all(filename)

    true_labels = np.array(true_labels)
    preds = np.array(preds)

    mask = true_labels != ""
    acc = np.mean(true_labels[mask] == preds[mask])
    print(f"Accuracy: {acc*100:.2f}%")
    return acc

if __name__ == "__main__":
    preds = predict_all("data/raw/training_data.csv")
    print(preds)
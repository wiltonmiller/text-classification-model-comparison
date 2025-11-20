import pandas as pd
import numpy as np
import json
import pickle
import re


def load_model_params(path="models/best_model_params.json"):
    with open(path, "r") as f:
        params = json.load(f)

    W = np.array(params["weights"])     # shape (num_classes, num_features)
    B = np.array(params["bias"])        # shape (num_classes,)
    class_ids = params["classes"]
    return W, B, class_ids


def load_label_classes(path="models/label_classes.json"):
    with open(path, "r") as f:
        return json.load(f)   # e.g. ["ChatGPT", "Claude", "Gemini"]


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


STOPWORDS = set([
    "the","a","an","and","or","in","on","at","to","is","are","was","were","be","been",
    "for","with","of","it","this","that","these","those","as","from","by","but","not"
])

def basic_clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)


def rename_columns(df):
    column_map = {
        'student_id': 'student_id',
        'In your own words, what kinds of tasks would you use this model for?': 'tasks_used_for',
        'How likely are you to use this model for academic tasks?': 'likelihood_academic',
        'Which types of tasks do you feel this model handles best? (Select all that apply.)': 'best_tasks_selected',
        'Based on your experience, how often has this model given you a response that felt suboptimal?': 'frequency_suboptimal',
        'For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)': 'suboptimal_tasks_selected',
        'Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?': 'suboptimal_response_details',
        'How often do you expect this model to provide responses with references or supporting evidence?': 'frequency_expected_refs',
        'How often do you verify this model\'s responses?': 'frequency_verification',
        'When you verify a response from this model, how do you usually go about it?': 'verification_method',
        'label': 'label'
    }
    return df.rename(columns=column_map)


def clean_dataframe(df_raw):
    df = df_raw.copy()

    # Rename columns
    df = rename_columns(df)

    # Fill NA text columns
    text_cols = ["tasks_used_for", "suboptimal_response_details", "verification_method"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
            df[c] = df[c].apply(basic_clean_text)

    # Fill NA frequency columns: replace NA with mode
    freq_cols = [
        "likelihood_academic",
        "frequency_suboptimal",
        "frequency_expected_refs",
        "frequency_verification"
    ]
    for c in freq_cols:
        if c in df.columns:
            mode = df[c].mode()[0]
            df[c] = df[c].fillna(mode).astype(str)

    # Strip numeric frequency answers
    for c in freq_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str[0].astype(int)

    # Drop student_id
    if "student_id" in df.columns:
        df = df.drop(columns=["student_id"])

    return df


def extract_features(df_clean):
    # Load artifacts saved during training
    with open("models/vectorizers.pkl", "rb") as f:
        vectorizers = pickle.load(f)

    with open("models/categorical_columns.json", "r") as f:
        select_all_schema = json.load(f)

    with open("models/feature_order.json", "r") as f:
        feature_order = json.load(f)

    df = df_clean.copy()

    # Remove label if present
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    # TF–IDF for text columns
    text_cols = ["tasks_used_for", "suboptimal_response_details", "verification_method"]
    for col in text_cols:
        if col not in df.columns:
            continue
        vec = vectorizers[col]
        transformed = vec.transform(df[col].astype(str))
        fnames = vec.get_feature_names_out()
        fnames = np.char.add(f"{col}_", fnames)
        tfidf_df = pd.DataFrame(transformed.toarray(), columns=fnames)
        df = pd.concat([df.drop(columns=[col]), tfidf_df], axis=1)

    # Normalize numeric frequency columns to [0,1]
    freq_cols = [
        "likelihood_academic",
        "frequency_suboptimal",
        "frequency_expected_refs",
        "frequency_verification"
    ]
    for col in freq_cols:
        if col in df.columns:
            df[col] = (df[col] - 1) / 4

    # Encode select-all columns
    select_cols = ["best_tasks_selected", "suboptimal_tasks_selected"]
    for col in select_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("").astype(str)
        df[col] = df[col].str.replace(r"\([^)]*\)", "", regex=True)
        df[col] = df[col].str.replace(" ", "_")
        df[col] = df[col].str.lower()
        df[col] = df[col].str.replace("/", "_")

        dummies = df[col].str.get_dummies(sep=",").add_prefix(f"{col}_")
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        # Ensure full schema exists
        if col in select_all_schema:
            for expected in select_all_schema[col]:
                if expected not in df:
                    df[expected] = 0

    # Align columns to training order
    for col in feature_order:
        if col not in df:
            df[col] = 0

    df = df[feature_order]

    return df.to_numpy().astype(float)

import pandas as pd
import numpy as np
import json
import re


def load_model_params(path="best_model_params.json"):
    with open(path, "r") as f:
        params = json.load(f)

    W = np.array(params["weights"])     # shape (num_classes, num_features)
    B = np.array(params["bias"])        # shape (num_classes,)
    class_ids = params["classes"]
    return W, B, class_ids


def load_label_classes(path="label_classes.json"):
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

def load_vectorizers_json(path="vectorizers.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def simple_tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()


def build_unigram_bigram_tokens(text):
    tokens = simple_tokenize(text)
    bigrams = [tokens[i] + " " + tokens[i+1] for i in range(len(tokens)-1)]
    return tokens + bigrams


def tfidf_from_json(series, vec_info):
    vocab = vec_info["vocabulary"]      # word -> index
    idf = np.array(vec_info["idf"])     # shape (n_features,)

    n_samples = len(series)
    n_features = len(idf)
    X_counts = np.zeros((n_samples, n_features))

    for i, text in enumerate(series):
        toks = build_unigram_bigram_tokens(text)
        for tok in toks:
            idx = vocab.get(tok)
            if idx is not None:
                X_counts[i, idx] += 1.0

    # term frequency
    row_sums = np.sum(X_counts, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    tf = X_counts / row_sums

    # tfidf = tf * idf
    tfidf = tf * idf

    # L2 normalization
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tfidf = tfidf / norms

    return tfidf


def make_tfidf_df(series, col_name, vec_info):
    vocab = vec_info["vocabulary"]
    idf = np.array(vec_info["idf"])

    # Correct index ordering
    n_features = len(idf)
    ordered_tokens = [None] * n_features
    for tok, idx in vocab.items():
        ordered_tokens[idx] = tok
    # Build TF-IDF matrix
    X_tfidf = tfidf_from_json(series, vec_info)

    # Create column names in the correct order
    colnames = [f"{col_name}_{tok}" for tok in ordered_tokens]

    return pd.DataFrame(X_tfidf, columns=colnames)


def encode_select_all(df, schema):
    for base_col, output_cols in schema.items():

        encoded = pd.DataFrame(0, index=df.index, columns=output_cols)

        raw = df[base_col].fillna("").astype(str)

        for i, val in raw.items():
            cleaned = re.sub(r"\([^)]*\)", "", val)
            cleaned = cleaned.replace(" ", "_").lower().replace("/", "_")
            pieces = [p.strip() for p in cleaned.split(",") if p.strip()]

            for p in pieces:
                colname = f"{base_col}_{p}"
                if colname in encoded.columns:
                    encoded.at[i, colname] = 1

        df = df.drop(columns=[base_col])
        df = pd.concat([df, encoded], axis=1)

    return df


def normalize_numeric(df):
    norm_cols = [
        "likelihood_academic",
        "frequency_suboptimal",
        "frequency_expected_refs",
        "frequency_verification",
    ]
    for c in norm_cols:
        df[c] = (df[c].astype(float) - 1.0) / 4.0
    return df


def extract_features(df_clean):

    # Load artifacts saved during training
    vectorizers = load_vectorizers_json("models/vectorizers.json")

    with open("models/categorical_columns.json", "r") as f:
        select_all_schema = json.load(f)

    with open("models/feature_order.json", "r") as f:
        feature_order = json.load(f)

    df = df_clean.copy()

    # Remove label for prediction
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    # Normalize numeric frequencies
    df = normalize_numeric(df)

    # Encode select-all columns
    df = encode_select_all(df, select_all_schema)

    # Manual TF–IDF for each text column
    text_cols = ["tasks_used_for", "suboptimal_response_details", "verification_method"]

    for col in text_cols:
        vec_info = vectorizers[col]
        tfidf_df = make_tfidf_df(df[col], col, vec_info)
        df = pd.concat([df.drop(columns=[col]), tfidf_df], axis=1)

    # Add missing columns & order correctly
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0.0

    df = df[feature_order]

    return df.to_numpy().astype(float)

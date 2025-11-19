import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
from ..utils import feature_helpers
import pickle, json


def extract_features(train_df, test_df):
    '''
    Extract features from cleaned dataframe
    
    Parameters:
    df: pandas.DataFrame
        Cleaned dataframe object
    
    Returns:
    numpy.ndarray
        Array of extracted features
    '''
    train_features, test_features = train_df.drop(['label'], axis=1), test_df.drop(['label'], axis=1)

    #convert text columns to tf idf vectors
    train_features, test_features, vectorizers = feature_helpers.vectorize_text(train_features, test_features)

    #normalise numeric columns to [0,1]
    train_features, test_features = feature_helpers.normalize_numeric(train_features), feature_helpers.normalize_numeric(test_features)

    #convert select all to binary indicators
    train_features, select_all_schema = feature_helpers.encode_select_all(train_features)
    test_features, _ = feature_helpers.encode_select_all(test_features)

    # makes sure that TEST has all TRAIN columns
    test_features = test_features.reindex(columns=train_features.columns, fill_value=0)

    feature_order = list(train_features.columns)
    return train_features, test_features, vectorizers, select_all_schema, feature_order


def extract_labels(df):
    '''
    Extract labels from cleaned dataframe
    
    Parameters:
    df: pandas.DataFrame
        Cleaned dataframe object
    
    Returns:
    numpy.ndarray
        Array of extracted labels
    '''
    encoder = LabelEncoder()

    label_encoded = encoder.fit_transform(df['label'])

    label_df = pd.DataFrame(label_encoded, columns=['label'])

    return label_df, encoder.classes_.tolist()

def save_to_npy(features, path):
    '''
    Save features as numpy array to specified path
    
    Parameters:
    features: numpy.ndarray
        Array of features to save
    path: string
        Output file path
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, features)
    print(f"Data saved to {path}")


if __name__ == "__main__":
    
    clean_training_data_path = "data/cleaned/training_data_clean.csv"
    clean_testing_data_path = "data/cleaned/testing_data_clean.csv"
    training_features_output_path_npy = "data/processed/training_features.npy"
    testing_features_output_path_npy = "data/processed/testing_features.npy"
    training_labels_output_path_npy = "data/processed/training_labels.npy"
    testing_labels_output_path_npy = "data/processed/testing_labels.npy"
    train_cleaned = pd.read_csv(clean_training_data_path)
    test_cleaned = pd.read_csv(clean_testing_data_path)
    
    # Extract and save features as well as processeing objects 
    train_features, test_features, vectorizers, select_all_schema, feature_order = extract_features(train_cleaned, test_cleaned)
    train_labels, label_classes = extract_labels(train_cleaned)
    
    save_to_npy(train_features.to_numpy(), training_features_output_path_npy)
    save_to_npy(test_features.to_numpy(), testing_features_output_path_npy)
    test_labels, _ = extract_labels(test_cleaned)

    save_to_npy(train_labels['label'].to_numpy(), training_labels_output_path_npy)
    save_to_npy(test_labels['label'].to_numpy(), testing_labels_output_path_npy)

    # save processing objects for later use
    # TF-IDF vectorizers
    with open("models/vectorizers.pkl", "wb") as f:
        pickle.dump(vectorizers, f)

    # schema for one-hot columns
    with open("models/categorical_columns.json", "w") as f:
        json.dump(select_all_schema, f, indent=2)

    # feature ordering
    with open("models/feature_order.json", "w") as f:
        json.dump(feature_order, f, indent=2)

    # classes in correct order (ChatGPT, Claude, Gemini)
    with open("models/label_classes.json", "w") as f:
        json.dump(label_classes, f, indent=2)

    print("Feature extraction + artifact saving successful")

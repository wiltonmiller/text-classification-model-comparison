import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
from ..utils import feature_helpers


def extract_features(df):
    '''
    Extract features from cleaned dataframe
    
    Parameters:
    df: pandas.DataFrame
        Cleaned dataframe object
    
    Returns:
    numpy.ndarray
        Array of extracted features
    '''
    features = df.drop(['label'], axis=1)

    #convert text columns to tf idf vectors
    features = feature_helpers.vectorize_text(features)

    #normalise numeric columns to [0,1]
    features = feature_helpers.normalize_numeric(features)

    #convert select all to binary indicators
    features = feature_helpers.encode_select_all(features)


    return features

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
    encoder = OneHotEncoder(sparse_output=False)

    label_encoded = encoder.fit_transform(df[['label']])

    feature_names = encoder.get_feature_names_out(['label'])
    # Result: ['label_ChatGPT', 'label_Claude', 'label_Gemini']

    label_df = pd.DataFrame(label_encoded, columns=feature_names)
    return label_df


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
    
    clean_data_path = "data/clean/training_data_clean.csv"
    features_output_path_csv = "src/data/processed/features.csv"
    features_output_path_npy = "src/data/processed/features.npy"
    labels_output_path_csv = "src/data/processed/labels.csv"
    labels_output_path_npy = "src/data/processed/labels.npy"
    df_cleaned = pd.read_csv(clean_data_path)
    
    # Extract and save features
    features = extract_features(df_cleaned)
    labels = extract_labels(df_cleaned)
    
    save_to_npy(features.to_numpy(), features_output_path_npy)
    features.to_csv(features_output_path_csv, index=False)

    save_to_npy(labels.to_numpy(), labels_output_path_npy)
    labels.to_csv(labels_output_path_csv, index=False)
    print("Feature extraction successful")

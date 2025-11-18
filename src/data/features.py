import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
from ..utils import feature_helpers


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
    train_features, test_features = feature_helpers.vectorize_text(train_features, test_features)

    #normalise numeric columns to [0,1]
    train_features, test_features = feature_helpers.normalize_numeric(train_features), feature_helpers.normalize_numeric(test_features)

    #convert select all to binary indicators
    train_features, test_features = feature_helpers.encode_select_all(train_features), feature_helpers.encode_select_all(test_features)


    return train_features, test_features

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
    
    clean_training_data_path = "data/cleaned/training_data_clean.csv"
    clean_testing_data_path = "data/cleaned/testing_data_clean.csv"
    training_features_output_path_npy = "data/processed/training_features.npy"
    testing_features_output_path_npy = "data/processed/testing_features.npy"
    training_labels_output_path_npy = "data/processed/training_labels.npy"
    testing_labels_output_path_npy = "data/processed/testing_labels.npy"
    train_cleaned = pd.read_csv(clean_training_data_path)
    test_cleaned = pd.read_csv(clean_testing_data_path)
    
    # Extract and save features
    train_features, test_features = extract_features(train_cleaned,test_cleaned)
    train_labels, test_labels = extract_labels(train_cleaned), extract_labels(test_cleaned)
    
    save_to_npy(train_features.to_numpy(), training_features_output_path_npy)
    save_to_npy(test_features.to_numpy(), testing_features_output_path_npy)

    save_to_npy(train_labels.to_numpy(), training_labels_output_path_npy)
    save_to_npy(test_labels.to_numpy(), testing_labels_output_path_npy)
    
    print("Feature extraction successful")

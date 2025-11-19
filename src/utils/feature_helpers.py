import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_text(train_df, test_df):
    text_cols = ['tasks_used_for','suboptimal_response_details','verification_method']
    #have to remove na again because converting to csv resets it
    train_df[text_cols] = train_df[text_cols].fillna("").astype(str)
    test_df[text_cols] = test_df[text_cols].fillna("").astype(str)

    # TF IDF Vectorization to be saved
    vectorizers = {}

    text_columns = {
        'tasks_used_for',
        'suboptimal_response_details',
        'verification_method'
    }
    
    for column in text_columns:
        tfidf = tfidf = TfidfVectorizer(
                            min_df=2,
                            ngram_range=(1, 2))
        vectorizers[column] = tfidf
        train_result = tfidf.fit_transform(train_df[column])
        test_result = tfidf.transform(test_df[column])
        feature_names = tfidf.get_feature_names_out()
        feature_names = np.char.add(f"{column}_", feature_names)
        train_X_df = pd.DataFrame(train_result.toarray(), columns=feature_names)
        train_X_df = train_X_df.round(4)
        train_df = pd.concat(
            [train_df.drop([column], axis=1),
            train_X_df],
            axis=1
        )

        test_X_df = pd.DataFrame(test_result.toarray(), columns=feature_names)
        test_X_df = test_X_df.round(4)
        test_df = pd.concat(
            [test_df.drop([column], axis=1),
            test_X_df],
            axis=1
        )
        print(f"Completed TF IDF processing for column {column}.")
    # need to also return the vectorizers to use later on test data
    return train_df, test_df, vectorizers


def normalize_numeric(df):
    numeric_columns = {
        'likelihood_academic',
        'frequency_suboptimal',
        'frequency_expected_refs',
        'frequency_verification'
    }

    for column in numeric_columns:
        df[column] = (df[column] - 1)/4
        print(f"Normalised column {column}.")
    
    print("Completed normalization.")

    return df




def encode_select_all(df):
    select_all_columns = {
        "best_tasks_selected",
        "suboptimal_tasks_selected",
    }
    
    schema = {}

    for column in select_all_columns:
        df[column] = df[column].str.replace(r"\([^)]*\)", "", regex=True)
        df[column] = df[column].str.replace(" ", "_")
        df[column] = df[column].str.lower()
        df[column] = df[column].str.replace("/", "_")
        encoded_columns = df[column].str.get_dummies(sep=",")
        encoded_columns = encoded_columns.add_prefix(f"{column}_")
        df = pd.concat(
            [df.drop([column], axis=1),
            encoded_columns],
            axis=1
        )
        # update schema with new columns
        if column not in schema:
            schema[column] = encoded_columns.columns.tolist()

        print(f"Completed encoding of column {column}.")
    print ("Completed select all encoding.")

    # return the schema for later use
    return df, schema
    
        
        

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_text(df):
    text_cols = ['tasks_used_for','suboptimal_response_details','verification_method']
    #have to remove na again because converting to csv resets it
    df[text_cols] = df[text_cols].fillna("").astype(str)
    text_columns = {
        'tasks_used_for',
        'suboptimal_response_details',
        'verification_method'
    }

    
    for column in text_columns:
        tfidf = TfidfVectorizer(min_df=3)
        result = tfidf.fit_transform(df[column])
        feature_names = tfidf.get_feature_names_out()
        feature_names = np.char.add(f"{column}_", feature_names)
        X_df = pd.DataFrame(result.toarray(), columns=feature_names)
        df = pd.concat(
            [df.drop([column], axis=1),
            X_df],
            axis=1
        )
        print(f"Completed TF IDF processing for column {column}.")
    return df


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
        print(f"Completed encoding of column {column}.")
    print ("Completed select all encoding.")
    return df
    
        
        

import pandas as pd
from ..utils import cleaning_helpers


'''
Load raw data from csv path into dataframe
Parameters:
path: string
    The path to the raw data file.

Returns:
pandas.DataFrame
    The dataframe holding the raw data.
'''
def load_data(path):
    print(f"Loading data from path {path}.")
    return pd.read_csv(path)


'''
Clean data and return cleaned dataframe

Parameters:
df: pandas.DataFrame
    Raw dataframe object

Returns:
pandas.DataFrame
    Cleaned dataframe object
'''
def clean_data(train_df, test_df):

    #rename columns
    train_df = cleaning_helpers.rename_columns(train_df)
    test_df = cleaning_helpers.rename_columns(test_df)

    #clear na
    train_df = cleaning_helpers.fill_na_text(train_df)
    test_df = cleaning_helpers.fill_na_text(test_df)

    #clean text(punctuation, lowercase, remove [model], lemmatize)
    train_df = cleaning_helpers.clean_text(train_df)
    test_df = cleaning_helpers.clean_text(test_df)
    
    #fill empty values
    train_df, test_df = cleaning_helpers.fill_na_frequency(train_df, test_df)
    


    #convert number scale to numbers
    train_df = cleaning_helpers.strip_frequencies(train_df)
    test_df = cleaning_helpers.strip_frequencies(test_df)
    train_df, test_df = train_df.drop(['student_id'], axis=1), test_df.drop(['student_id'], axis=1)
    return train_df, test_df





'''
Saves the cleaned data as a csv file at the location specified by path

Parameters:
df: pandas.DataFrame
    Stores the cleaned dataframe object.
path: string
    Export path.
'''
def save_data(df, path):
    print(f"Saving cleaned data to {path}.")
    df.to_csv(path, index=False)


def split_data(df):
    unique_ids = df['student_id'].drop_duplicates().tolist()

    # 2. Determine split index
    split_idx = int(len(unique_ids) * 0.8)

    # 3. Split student IDs
    train_ids = unique_ids[:split_idx]
    test_ids  = unique_ids[split_idx:]

    # 4. Assign rows to train/test based on student_id
    train_df = df[df['student_id'].isin(train_ids)]
    test_df  = df[df['student_id'].isin(test_ids)]
    return train_df, test_df


if __name__ == "__main__":

    
    raw_data_path = "data/raw/training_data.csv"
    clean_data_path_train = "data/cleaned/training_data_clean.csv"
    clean_data_path_test = "data/cleaned/testing_data_clean.csv"
    df_raw = load_data(raw_data_path)
    #split the data
    train_raw, test_raw = split_data(df_raw)

    train_cleaned, test_cleaned = clean_data(train_raw, test_raw)
    save_data(train_cleaned, clean_data_path_train)
    save_data(test_cleaned, clean_data_path_test)
    print("Data cleaning successful")





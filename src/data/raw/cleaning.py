import pandas as pd
from ...utils import cleaning_helpers


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
def clean_data(df):

    #rename columns
    df = cleaning_helpers.rename_columns(df)

    #fill empty values
    df = cleaning_helpers.fill_na_text(df)
    
    
    #clean text(punctuation, lowercase, remove [model], lemmatize)
    df = cleaning_helpers.clean_text(df)
    
    #fill empty values
    df = cleaning_helpers.fill_na_frequency(df)


    #convert number scale to numbers
    df = cleaning_helpers.strip_frequencies(df)

    return df





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




if __name__ == "__main__":
    raw_data_path = "data/raw/training_data.csv"
    clean_data_path = "data/clean/training_data_clean.csv"
    df_raw = load_data(raw_data_path)
    df_cleaned = clean_data(df_raw)
    save_data(df_cleaned, clean_data_path)
    print("Data cleaning successful")





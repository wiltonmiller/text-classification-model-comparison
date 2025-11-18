from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
import spacy

#for lemmatization
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


def rename_columns(df):
    new_column_names = {
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

    # Apply the rename operation
    df.rename(columns=new_column_names, inplace=True)
    print("Columns have been renamed.")
    return df


def fill_na_frequency(train_df, test_df):
    print("------------------\nSTARTING NA FILLING FOR FREQUENCY COLUMNS\n------------------")
    columns_to_fill = {
        'likelihood_academic',
        'frequency_suboptimal',
        'frequency_expected_refs',
        'frequency_verification'
    }
    for column in columns_to_fill:
        mode = train_df[column].mode()[0]
        #Fill blanks with the mode of that column
        train_df.fillna({column:mode}, inplace=True)
        test_df.fillna({column:mode}, inplace=True)

        print(f"Completed cleaning column {column}.")
    
    print("------------------\nCOMPLETED NA FILLING FOR FREQUENCY COLUMNS\n------------------")
    return train_df, test_df

def fill_na_text(df):
    print("------------------\nSTARTING NA FILLING FOR TEXT COLUMNS\n------------------")
    columns_to_fill = {
        'tasks_used_for',
        'suboptimal_response_details',
        'verification_method'
    }
    for column in columns_to_fill:
        df.fillna({column:""}, inplace=True)
        print(f"Completed cleaning column {column}.")
    
    print("------------------\nCOMPLETED NA FILLING FOR TEXT COLUMNS\n------------------")
    return df


def clean_text(df):
    print("------------------\nSTARTING TEXT CLEANING\n------------------")
    #remove #NAME?, placeholders from data
    df.replace(r"\[THIS MODEL\]", "this model", regex=True, inplace=True)
    df.replace(r"\[ANOTHER MODEL\]", "another model", regex=True, inplace=True)
    df.replace(r"#NAME\?", "", regex=True, inplace=True)    
    print("Placeholders have been cleared.")

    #convert to lowercase, remove punctuation, strip whitespace, remove stop words
    columns_to_lower = {
        'tasks_used_for',
        'suboptimal_response_details',
        'verification_method'
    }

    stop_words = ENGLISH_STOP_WORDS

    for column in columns_to_lower:
        #lowercase, strip and replace punctuation and numbers
        df[column] = df[column].str.lower()
        df[column] = df[column].str.replace("'", "")
        df[column] = df[column].str.replace("’", "")
        df[column] = df[column].str.replace(r"[^a-zA-Z\s]", " ", regex=True)
        df[column] = df[column].str.strip()
        #remove stop words
        df[column] = df[column].apply(
            lambda x: " ".join([word for word in x.split() if word not in stop_words])
        )
       

        #lemmatize
        df[column] = df[column].apply(lemmatize_text_spacy)

        #replace cod with code
        df[column] = df[column].str.replace(r"\bcod\b", "code", regex=True)
        print(f"Completed cleaning column {column}.")
    print("------------------\nCOMPLETED TEXT CLEANING\n------------------")
    return df
    



def strip_frequencies(df):
    print("------------------\nSTARTING INTEGER STRIPPING\n------------------")
    columns_to_strip = {
        'likelihood_academic',
        'frequency_suboptimal',
        'frequency_expected_refs',
        'frequency_verification'
    }

    for column in columns_to_strip:
        df[column] = df[column].str[0].astype(int)
        print(f"Completed integer stripping for column {column}.")
    
    print("------------------\nCOMPLETED INTEGER STRIPPING\n------------------")
    return df


def lemmatize_text_spacy(text):
    if pd.isna(text):
        return text
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])
    


    




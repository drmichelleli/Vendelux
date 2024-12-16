import pandas as pd
import re
def merge_text(row):
    if pd.isnull(row['SHARED_POST_TEXT']) and ~pd.isnull(row['POST_TEXT']):
        return 'post date: '+str(row['post_date']) + ' '+ str(row['POST_TEXT'])
    elif ~pd.isnull(row['SHARED_POST_TEXT']) and pd.isnull(row['POST_TEXT']):
        return 'post date: '+str(row['post_date']) + ' '+ str(row['SHARED_POST_TEXT'])
    else:
        return 'post date: '+str(row['post_date']) +  ' ' + str(row['POST_TEXT'])+ ' ' + str(row['SHARED_POST_TEXT'])
    
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Remove extra whitespace
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    return text

def filter_data(input_df, event_date = '2024-09-18', months_before = -12, months_after = 12):
    input_df['event_date'] = event_date
    input_df['event_date'] = pd.to_datetime(input_df['event_date'])
    input_df['post_date'] = input_df['DATE_PUBLISHED'].str[0:10]
    input_df['post_date'] = pd.to_datetime(input_df['post_date'])
    input_df['months_diff'] = (input_df['event_date'] - input_df['post_date']).dt.days/30
    data_df =input_df[(input_df['months_diff'] >= months_before) & (input_df['months_diff'] <= months_after)]
    return data_df
from libs.event_search_func import KeyWordSearch
from datetime import datetime
import logging
import pandas as pd
from libs.preprocess_func import merge_text, clean_text, filter_data

logging.basicConfig(filename='event_search_'+datetime.now().strftime('%Y_%m_%d_%H_%M')+'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# load the data and remove duplicates
data_df = pd.read_excel('./input/Dataset.xlsx', sheet_name='post_details')
data_df.drop_duplicates(subset='POST_ID', inplace=True)

# create the string field for the post_date 
data_df['post_date'] = data_df['DATE_PUBLISHED'].str[0:10]
data_df['post_date'] = data_df['post_date'].astype(str)

#merge the 'POST_TEXT', 'SHARED_POST_TEXT' and 'post_date' columns
data_df['text'] = data_df.apply(merge_text, axis=1)

# Apply the clean_text function to the 'text' column
data_df['cleaned_text'] = data_df['text'].apply(clean_text)
logging.info("input data size: %s",len(data_df))

# start the key word search algorithm on the whole data set
logging.info("Key word search on the whole data set ")
event_search_func = KeyWordSearch(data_df,'dmexco')
whole_output_df = event_search_func.run()
whole_output_df.to_csv('output/event_key_word_whole_data_set.csv', index=False)

# start the key word search algorithm on the filtered data set
logging.info("Key word search on the filtered data set ")
filtered_df = filter_data(data_df, event_date='2024-09-18', months_before=-12, months_after=12)
logging.info("filtered data size: %s",len(filtered_df))
event_search_func = KeyWordSearch(filtered_df,'dmexco')
filltered_output_df = event_search_func.run()
filltered_output_df.to_csv('output/event_key_word_filtered_data_set.csv', index=False)





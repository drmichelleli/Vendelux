import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import logging
from datetime import datetime, timedelta
from libs.preprocess_func import merge_text, clean_text, filter_data
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set the start time
start = datetime.now()

# Set up the logging configuration
logging.basicConfig(filename='key_word_search.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the input data
input_df = pd.read_excel('input/Dataset.xlsx', sheet_name='post_details')
# Drop duplicates based on the 'POST_ID' column
input_df.drop_duplicates(subset='POST_ID', inplace=True)
logging.info("input data size: %s", len(input_df))

# Filter the data to include only the posts within 12 months before or after the event date
# the event date, months_before and months_after can be changed to any date and months
data_df = filter_data(input_df, event_date='2024-09-18', months_before=-12, months_after=12)
logging.info("filtered data size: %s", len(data_df))

#merge the 'POST_TEXT', 'SHARED_POST_TEXT' and 'post_date' columns
data_df['text'] = data_df.apply(merge_text, axis=1)

# Apply the clean_text function to the 'text' column
data_df['cleaned_text'] = data_df['text'].apply(clean_text)

# Define a function to search for a specific event keyword in the text. In this case, we are searching for 'dmexco'. 
# the event name can be replace by any event name
def search_event(text, event_name = 'dmexco'):
    return 'yes' if event_name in text else 'no'

# Apply the search_dmexco function to the 'cleaned_text' column
data_df['event_found'] = data_df['cleaned_text'].apply(search_event)
data_df.to_csv('output/event_key_word.csv', index=False)

# Calculate accuracy
true_values = data_df['IS_ATTENDING']
predicted_values = data_df['event_found']
accuracy = accuracy_score(true_values, predicted_values)
logging.info(f'Accuracy: {accuracy}')


# Print classification report
class_report = classification_report(true_values, predicted_values)
logging.info('Classification Report:')
logging.info(class_report)

end = datetime.now()
time_taken = pd.Timedelta(end-start, unit = 'm')/timedelta(minutes=1)
logging.info('finished successfully and it took ' + str(time_taken) + ' minutes')
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import logging
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings('ignore') 

class KeyWordSearch:
    def __init__(self,data_df, event_name):
        self.data_df = data_df
        self.event_name = event_name

    # Define a function to search for a specific event keyword in the text. 
    def search_event(self, text):
        return 'yes' if self.event_name in text else 'no'
    
    # Run the key word search algorithm
    def run(self):
        start = datetime.now()
        # Apply the search_event function to the 'cleaned_text' column
        self.data_df['event_found'] = self.data_df['cleaned_text'].apply(self.search_event)

        # Calculate accuracy
        true_values = self.data_df['IS_ATTENDING']
        predicted_values = self.data_df['event_found']
        accuracy = accuracy_score(true_values, predicted_values)
        logging.info(f'Accuracy: {accuracy}')

        # Print classification report
        class_report = classification_report(true_values, predicted_values)
        logging.info('Classification Report:')
        logging.info(class_report)

        end = datetime.now()
        time_taken = pd.Timedelta(end-start, unit = 'm')/timedelta(minutes=1)
        logging.info('"Key word search algorithm finished successfully and it took ' + str(time_taken) + ' minutes\n')
        return self.data_df






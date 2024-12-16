from datetime import datetime
import logging
import pandas as pd
from libs.preprocess_func import merge_text, filter_data
from libs.nn_classifier_func import NNClassifier

logging.basicConfig(filename='nn_classifier_'+datetime.now().strftime('%Y_%m_%d_%H_%M')+'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# load the data and remove duplicates
data_df = pd.read_excel('./input/Dataset.xlsx', sheet_name='post_details')
data_df.drop_duplicates(subset='POST_ID', inplace=True)

# create the string field for the post_date
data_df['post_date'] = data_df['DATE_PUBLISHED'].str[0:10]
data_df['post_date'] = data_df['post_date'].astype(str)

#merge the 'POST_TEXT', 'SHARED_POST_TEXT' and 'post_date' columns
data_df['text'] = data_df.apply(merge_text, axis=1)

logging.info("input data size: %s",len(data_df))

# start the neural network classifier on the whole data set
logging.info("Neural network classifier on the whole data set ")
nn_func = NNClassifier(data_df)
whole_output_df = nn_func.run()
whole_output_df.to_csv('output/nn_classifier_whole_data_set.csv', index=False)

# start the key word search algorithm on the filtered data set
logging.info("Neural network classifier on the filtered data set ")
filtered_df = filter_data(data_df, event_date='2024-09-18', months_before=-12, months_after=12)
logging.info("filtered data size: %s",len(filtered_df))
nn_func = NNClassifier(filtered_df)
filltered_output_df = nn_func.run()
filltered_output_df.to_csv('output/nn_classifier_filtered_data_set.csv', index=False)
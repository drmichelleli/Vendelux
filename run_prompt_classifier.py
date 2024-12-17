from datetime import datetime, timedelta
import logging
import asyncio
import pandas as pd
from libs.preprocess_func import merge_text, filter_data
from libs.prompt import event_prompt
from libs.llm_func import LLMScore
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')


logging.basicConfig(filename='prompt_classifier_'+datetime.now().strftime('%Y_%m_%d_%H_%M')+'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

start = datetime.now()
# load the data and remove duplicates
data_df = pd.read_excel('Vendelux - Sr. Data Scientist Take Home Assessment - Dataset (1) (1).xlsx', sheet_name='post_details')
data_df.drop_duplicates(subset='POST_ID', inplace=True)

# create the string field for the post_date
data_df['post_date'] = data_df['DATE_PUBLISHED'].str[0:10]
data_df['post_date'] = data_df['post_date'].astype(str)

#merge the 'POST_TEXT', 'SHARED_POST_TEXT' and 'post_date' columns
data_df['text'] = data_df.apply(merge_text, axis=1)
data_df['id'] = data_df['POST_ID']
logging.info("input data size: %s",len(data_df))

# start the prompt classifier on the whole data set
logging.info("prompt classifier on the whole data set")
classifier = LLMScore(data_df, event_prompt)
asyncio.run(classifier.run())
responses = classifier.return_results()
chpt_classified_data = pd.DataFrame(responses)
output_df = pd.merge(data_df, chpt_classified_data, on = 'id', how ='inner')
end = datetime.now()
time_taken = pd.Timedelta(end-start, unit='m') / timedelta(minutes=1)
logging.info(f'Prompt classifier finished successfully and it took {time_taken} minutes')

# print the accuracy
true_values = output_df['IS_ATTENDING']
predicted_values = output_df['event_found']
accuracy = accuracy_score(true_values, predicted_values)
logging.info(f'whole data set accuracy: {accuracy:.4f}')

# Print classification report
class_report = classification_report(true_values, predicted_values)
logging.info(f'whole data set classification Report:\n{class_report}')

# save the output
output_df.drop(columns=['id', 'text'], inplace=True)
output_df.to_csv('./output/prompt_classifier_whole_data_set.csv', index=False)

# start the prompt classifier on the filtered data set
start = datetime.now()
logging.info("prompt classifier on the filtered data set")
filtered_df = filter_data(data_df, event_date='2024-09-18', months_before=-12, months_after=12)
logging.info("filtered data size: %s",len(filtered_df))

classifier = LLMScore(filtered_df, event_prompt)
asyncio.run(classifier.run())
responses = classifier.return_results()
chpt_classified_data = pd.DataFrame(responses)
output_df = pd.merge(filtered_df, chpt_classified_data, on = 'id', how ='inner')

end = datetime.now()
time_taken = pd.Timedelta(end-start, unit='m') / timedelta(minutes=1)
logging.info(f'Prompt classifier finished successfully and it took {time_taken} minutes')

# print the accuracy
true_values = output_df['IS_ATTENDING']
predicted_values = output_df['event_found']
accuracy = accuracy_score(true_values, predicted_values)
logging.info(f'filtered data set accuracy: {accuracy:.4f}')

# Print classification report
class_report = classification_report(true_values, predicted_values)
logging.info(f'filtered data set classification Report:\n{class_report}')
output_df.drop(columns=['id', 'text'], inplace=True)
output_df.to_csv('./output/prompt_classifier_filtered_data_set.csv', index=False)
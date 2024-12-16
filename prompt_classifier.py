import pandas as pd
from datetime import datetime, timedelta
from libs.llm_fun import LLMScore 
import asyncio
from libs.preprocess_func import merge_text
from libs.prompt import event_prompt
from sklearn.metrics import accuracy_score, classification_report

start = datetime.now()

# prepare the training and testing data for the classification model
input_df = pd.read_excel('input/Dataset.xlsx', sheet_name='post_details')
input_df.drop_duplicates(subset = 'POST_ID', inplace = True)

input_df['post_date'] = input_df['DATE_PUBLISHED'].str[0:10]
input_df['post_date'] = input_df['post_date'].astype(str)

input_df['text'] = input_df.apply(merge_text, axis=1)
input_df['id'] = input_df['POST_ID']
print('input file length', len(input_df))

classifier = LLMScore(input_df, event_prompt)
asyncio.run(classifier.run())
responses = classifier.return_results()
chpt_classified_data = pd.DataFrame(responses)

output_df = pd.merge(input_df, chpt_classified_data, on = 'id', how ='inner')
output_df.to_csv('output/prompt_classification_output.csv', index = False)
print('output file length', len(output_df))

true_values = output_df['IS_ATTENDING']
predicted_values = output_df['dmexco_found']
accuracy = accuracy_score(true_values, predicted_values)
print(f'Accuracy: {accuracy}')

# Print classification report
class_report = classification_report(true_values, predicted_values)
print('Classification Report:')
print(class_report)

end = datetime.now()
time_taken = pd.Timedelta(end-start, unit = 'm')/timedelta(minutes=1)
print('finished successfully and it took ' + str(time_taken) + ' minutes')
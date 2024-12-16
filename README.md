# Vendelux

The Vendelux project aims to determine whether LinkedIn posts indicate attendance at an event. The task is to classify posts that may or may not be related to an event named **DMEXCO 2024**. The goal is to rapidly develop a proof of concept method that is “good enough” to be used for classifying whether a post suggests the poster is attending the event.

## Dataset 

The dataset consists of approximately 1,600 labeled observations. Each observation contains a LinkedIn post and a label:

- **Yes**: The post suggests the user is attending the 2024 event (even loosely, such as planning to attend, promoting, or expressing general interest).
- **No**: The post does not suggest the user is attending the 2024 event.


## Method

Three methods are provided in the repository. 
 - **1. Event search** 
       To run this algorithm: 
       - First, activate the environment: source .venv/bin/activate
       - Secondly, run the code: python run_event_search.py
       - outputs
            - one log file: event_search_YYYY_MM_DD_HH_MM.log
            - two csv files: ./output/event_key_word_filtered_data_set.csv  and ./output/event_key_word_whole_data_set.csv

 - **2. Text embedding + neural network classifier**
        To run this algorithm:
       - First, activate the environment: source .venv/bin/activate
       - Secondly, run the code: python run_nn_classifier.py
       - outputs
            - one log file: nn_classifier_YYYY_MM_DD_HH_MM.log
            - two csv files: ./output/nn_classifier_filtered_data_set.csv  and ./output/nn_classifier_whole_data_set.csv

- **3. Prompt engineering with LLM**
      To run this algorithm: 
      - First, provide openai API key in the libs/llm_fun.py   
      - Secondly, activate the environment: source .venv/bin/activate
      - - outputs
            - one log file: nn_classifier_YYYY_MM_DD_HH_MM.log
            - two csv files: ./output/prompt_classifier_filtered_data_set.csv  and ./output/prompt_classifier_whole_data_set.csv
- 

# Vendelux

The Vendelux project aims to determine whether LinkedIn posts indicate attendance at an event. The task is to classify posts that may or may not be related to an event named **DMEXCO 2024**. The goal is to rapidly develop a proof of concept method that is “good enough” to be used for classifying whether a post suggests the poster is attending the event.

## Dataset

The dataset consists of approximately 1,600 labeled observations. Each observation contains a LinkedIn post and a label:

## Labels

- **Yes**: The post suggests the user is attending the 2024 event (even loosely, such as planning to attend, promoting, or expressing general interest).
- **No**: The post does not suggest the user is attending the 2024 event.

## Model Training and Evaluation

Three methods were provided inthe repository

### 1. Key word search

To run the key word search
1. Activate the environment:
    ```sh
    source .venv/bin/activate
    ```
2. Execute the script:
    ```sh
    python nn_classifier.py

    ```
The outputs include:
    - Log file: `event_search_YYYY_MM_DD_HH_MM.log`
    - CSV files: `./output/event_key_word_filtered_data_set.csv` and `./output/event_key_word_whole_data_set.csv`


### 2. Neural Network Classifier

To run the neural network classifier:

1. Activate the environment:
    ```sh
    source .venv/bin/activate
    ```
2. Execute the script:
    ```sh
    python nn_classifier.py
    ```
The outputs include:
    - Log file: `nn_classifier_YYYY_MM_DD_HH_MM.log`
    - CSV files: `./output/nn_classifier_filtered_data_set.csv` and `./output/nn_classifier_whole_data_set.csv`

### 3. Prompt Engineering with LLM

To run the prompt engineering algorithm:

1. Provide the OpenAI API key in `libs/llm_fun.py`.
2. Activate the environment:
    ```sh
    source .venv/bin/activate
    ```
3. Execute the script:
    ```sh
    python prompt_classifier.py
    ```
The outputs include:
    - Log file: `prompt_classifier_YYYY_MM_DD_HH_MM.log`
    - CSV files: `./output/prompt_classifier_filtered_data_set.csv` and `./output/prompt_classifier_whole_data_set.csv`

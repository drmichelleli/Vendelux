# Vendelux

The Vendelux project aims to determine whether LinkedIn posts indicate attendance at an event. The task is to classify posts that may or may not be related to an event named **DMEXCO 2024**. The goal is to rapidly develop a proof of concept method that is “good enough” to be used for classifying whether a post suggests the poster is attending the event.

## Dataset 

The dataset consists of approximately 1,600 labeled observations. Each observation contains a LinkedIn post and a label:

- **Yes**: The post suggests the user is attending the 2024 event (even loosely, such as planning to attend, promoting, or expressing general interest).
- **No**: The post does not suggest the user is attending the 2024 event.


## Method

Three methods are provided in the repository. 
 1. key word search 
 2. Text embedding + neural network classifier
 3. Prompt engineering with LLM

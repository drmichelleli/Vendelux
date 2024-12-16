def event_prompt(text):
    prompt = f""" you are given a social media post with the post date. The task is to determine if the post is related to an event named DMEXCO 2024. \
                Such as planning to attend, promoting, expressing general interest, attending experience, advertising event features etc. \
                The event date is September 18, 2024. Compare the post date with the event date to make a decision if the post is related to the 2024 event DMEXCO.\
                The post date needs to within 12 months before or after the event date. \
                Make sure that the post about the DMEXCO in 2024 not in other years. return 'yes' if the post is related to the DMEXCO 2024, return 'no' if not.\
                Also output the reason why you make this judgement. \
                The output should be in json format with the keys being "dmexco_found" and "reason". Make sure the json format is correct and \
                can be parsed by a json.loads function \
                the social media post is given below delimited by triple backticks.\
             ```{text}```
        """
    return prompt
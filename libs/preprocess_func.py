import pandas as pd
def merge_text(row):
    if pd.isnull(row['SHARED_POST_TEXT']) and ~pd.isnull(row['POST_TEXT']):
        return 'post date: '+str(row['post_date']) + ' '+ str(row['POST_TEXT'])
    elif ~pd.isnull(row['SHARED_POST_TEXT']) and pd.isnull(row['POST_TEXT']):
        return 'post date: '+str(row['post_date']) + str(row['SHARED_POST_TEXT'])
    else:
        return 'post date: '+str(row['post_date']) + str(row['POST_TEXT'])+ ' ' + str(row['SHARED_POST_TEXT'])
#%%imports
import pandas as pd

#%%data paths
train_path = '/home/ugoekay/Desktop/kaggle_disaster/nlp-getting-started/train.csv'
test_path = '/home/ugoekay/Desktop/kaggle_disaster/nlp-getting-started/test.csv'
target_path = '/home/ugoekay/Desktop/kaggle_disaster/nlp-getting-started/sample_submission.csv'

train_output = '/home/ugoekay/Desktop/kaggle_disaster/nlp-getting-started/processed/train.csv'
test_output = '/home/ugoekay/Desktop/kaggle_disaster/nlp-getting-started/processed/test.csv'

#%%read data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
target_df = pd.read_csv(target_path)

# %%merge target and test file
df_merged = pd.merge(test_df,target_df,on = 'id', how='left')

# %% drop unnecessary cols
train_df = train_df.drop(columns=['id','keyword', 'location'])
df_merged = df_merged.drop(columns=['id','keyword', 'location'])

#write to csv file
train_df.to_csv(train_output, sep='\t', index=False)
df_merged.to_csv(test_output, sep='\t', index=False)
# %%

import torch
from torch.utils.data import Dataset, DataLoader

#%% create pythorch dataset

#extend torch dataset -->  to efficiently load the data

class TwitterDataset(Dataset):

  """
  Characterizes a dataset for PyTorch


  Parameters
  ----------
    data : df
        tweets and sentiments in dataframe
    tokenizer :PreTrainedTokenizer
        tokenizer to tokenize the texts
    max_token_len : int
        maximum length of the tokenized tweets

  Returns
  --------
  dictionary

      text : str
          text to evaluate results further
      input_ids : tensor
          corresponding ids of tokenized words (lenght = max_len) 
      attention_mask:tensor
          list of importance of the tokenized word(attention map)(lenght = max_len) 
      targets: tensor
          targets that given tweets have whether disaster exist or not
  """
  def __init__(self, data, tokenizer, max_token_len):
    
    self.tokenizer = tokenizer
    self.data = data
    self.max_token_len = max_token_len
    
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, index):

    data_row = self.data.iloc[index]
    text = data_row.text
    target = data_row.target


    # we will be using encode plus method to tokenization and creation of pytorch tensor
    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,       # Add '[CLS]' and '[SEP]'
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding='max_length' ,        # Add Padding
      truncation=True , 
      return_attention_mask=True,    
      return_tensors='pt',           # Return Pytorch tensor 
    )


    return {
      'texts': text,    #to evaluate predections         
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)}

#imports
import pandas as pd
import numpy as np

import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from dataset import TwitterDataset

#%%create a data module
class TwitterDataModule(pl.LightningDataModule):

  def __init__(self, train_path:str, test_path:str, model_name:str ,batch_size: int, max_len:int, num_workers: int = 4, random_seed:int = 42):
    super().__init__()

    self.train_path = train_path
    self.test_path = test_path
    self.model_name = model_name
    self.batch_size = batch_size
    self.max_token_len = max_len
    self.num_workers = num_workers
    self.random_seed = random_seed

    #read data  
    self.train_df = pd.read_csv(self.train_path, sep='\t')
    self.test_df = pd.read_csv(self.test_path, sep='\t')
    self.class_names = list(set(self.train_df['target']))
    self.n_classes = len(self.class_names)
    self.len_train = len(self.train_df)

    ###class dict
    #class_weights = compute_class_weight(class_weight = "balanced", classes= np.unique(self.train_df['target']), y= self.train_df['target'])
    #self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(torch.device('cuda'))
    ###class dict
    

  
  def setup(self, stage=None):

    #init tokenizer    
    tokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length=self.max_token_len)

    self.train_dataset = TwitterDataset(self.train_df, tokenizer,self.max_token_len)
    self.test_dataset = TwitterDataset(self.test_df, tokenizer, self.max_token_len)

  
  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

  def val_dataloader(self):
    return DataLoader(self.test_dataset,batch_size=self.batch_size,num_workers=self.num_workers)

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
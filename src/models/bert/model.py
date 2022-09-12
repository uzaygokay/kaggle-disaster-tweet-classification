#%% imports
import plotly.figure_factory as ff

from transformers import BertModel
import torch.nn as nn
import torch

import pytorch_lightning as pl

from torchmetrics import MetricCollection, Accuracy, Precision, Recall,  F1Score
#from mlflow.tracking import MlflowClient

from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

#%% create sentiment classifier
class TextClassifier(pl.LightningModule):

  #constructor accepts number of classes we have
  def __init__(self, model_name, n_classes, class_names, dropout, learning_rate, weight_decay, epsilon, n_training_steps=None, n_warmup_steps=None, class_weights=None):   
    super(TextClassifier, self).__init__()

    #save hyperparameters
    self.save_hyperparameters()

    #attributes
    self.model_name = model_name
    self.n_classes = n_classes
    self.class_names = class_names
    self.dropout = dropout
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    self.epsilon = epsilon
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    self.class_weights = class_weights
    
    #initialize BERT model
    self.bert = BertModel.from_pretrained(self.model_name)
    self.drop = nn.Dropout(p=self.dropout)
    self.out = nn.Linear(self.bert.config.hidden_size, self.n_classes)

    
    metrics = MetricCollection({
      'accuracy': Accuracy(), 
      'f1_micro' : F1Score(num_classes=self.n_classes, average='micro'),
      'F1Score' : F1Score(num_classes=self.n_classes, average=None),
    },  ) #compute_groups=True


    self.train_metrics = metrics.clone(prefix='train_')
    self.valid_metrics = metrics.clone(prefix='val_')
    self.test_metrics = metrics.clone(prefix='test_')

  def forward(self, input_ids, attention_mask):

    _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
    output = self.drop(pooled_output)
    outputs = self.out(output)

    return outputs


  def __single_step(self, batch) :
    
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    targets = batch["targets"]

    # Forward pass
    outputs = self(input_ids=input_ids,attention_mask=attention_mask)

    return targets, outputs


  def training_step(self, batch, batch_idx):

    # labels and outputs for each batch
    targets, outputs = self.__single_step(batch)
    
    #calculate loss and update metrics
    loss_fn = nn.CrossEntropyLoss() #weight=self.class_weights
    loss = loss_fn(outputs, targets)
    self.train_metrics.update(outputs,targets)
    
    #log training loss
    self.log("train_loss", loss, on_step=False, on_epoch=True)
    return {"loss": loss}

  
  def training_epoch_end(self, outputs):

    #compute metrics and reset
    result = self.train_metrics.compute()
    self.train_metrics.reset()

    #log metrics
    self.log_dict(result, on_epoch=True)


  def validation_step(self, batch, batch_idx):

    # labels and outputs for each batch
    targets, outputs = self.__single_step(batch)
    
    #calculate loss and update metrics
    loss_fn = nn.CrossEntropyLoss()#weight=self.class_weights
    loss = loss_fn(outputs, targets)
    self.valid_metrics.update(outputs,targets)

    #log validation loss
    self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
    return {"val_loss": loss}


  def validation_epoch_end(self, outputs):
    
    #compute metrics and reset
    result = self.valid_metrics.compute()
    self.valid_metrics.reset()

    #log metrics
    self.log_dict(result, on_epoch=True, prog_bar=True)

  
  def test_step(self, batch, batch_idx):

    # labels and outputs for each batch
    targets, outputs = self.__single_step(batch)

    #calculate loss and update metrics
    loss_fn = nn.CrossEntropyLoss() #weight=self.class_weights
    loss = loss_fn(outputs, targets)
    self.test_metrics.update(outputs,targets)

    #log test loss
    self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
    return {"test_loss": loss}

  
  def test_epoch_end(self, outputs):
    
    #compute metrics and reset
    result = self.test_metrics.compute()
    self.test_metrics.reset()

    #log metrics
    self.log_dict(result , on_epoch=True)
    
  
  def configure_optimizers(self):
    
    #optimizer
    optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, eps=self.epsilon)

    #scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.n_warmup_steps, num_training_steps=self.n_training_steps)

    return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler,interval='step'))
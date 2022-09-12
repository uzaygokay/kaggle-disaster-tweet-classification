#%% imports
import warnings

from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.loggers import MLFlowLogger

import config
from arguments import get_args
from dataloader import TwitterDataModule
from logger import get_custom_logger
from model import TextClassifier

#get data and model names
TRAIN_PATH = config.TRAIN_PATH
TEST_PATH = config.TEST_PATH
MODEL_NAME = config.PRE_TRAINED_MODEL_NAME


if __name__ == '__main__':
  
  #arg and logger
  args = get_args()
  logger = get_custom_logger(__name__)
    
  #ignore some of the warnings
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

  # Initialize data module
  logger.info("Init DataModule")
  dm = TwitterDataModule(
    train_path=TRAIN_PATH, 
    test_path=TEST_PATH, 
    model_name=MODEL_NAME,
    batch_size=args.batch_size,
    max_len=args.max_length,
    num_workers = args.num_workers,
    random_seed = args.random_seed
    )

  #Initialize model
  logger.info("Init model from datamodule's attributes")
  steps_per_epoch= dm.len_train // args.batch_size
  total_training_steps = steps_per_epoch * args.max_epochs
  warmup_steps = total_training_steps // 5
  
  model = TextClassifier(
    MODEL_NAME,
    dm.n_classes,
    dm.class_names, 
    learning_rate=args.learning_rate,
    dropout = args.dropout,  
    weight_decay=args.weight_decay,
    epsilon=args.epsilon,
    n_training_steps=total_training_steps, 
    n_warmup_steps=warmup_steps,
    #class_weights = dm.class_weights
  )

  #callbacks
  logger.info("Init callbacks")
  callbacks = []
  progressbar_callback = RichProgressBar(refresh_rate=1, leave=True)
  callbacks.append(progressbar_callback)

  #mlflow logger
  #mlflow_logger = MLFlowLogger(experiment_name=STUDY_NAME, tracking_uri="file:./mlflow-runs")
  # execute to see results --> mlflow ui --backend-store-uri file://`pwd`/mlflow-runs

  #Initialize trainer
  logger.info("Init trainer")
  trainer = Trainer(
        #fast_dev_run=7,
        gpus=args.gpus,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        max_epochs=args.max_epochs,
        enable_progress_bar=True,
        callbacks=callbacks,
        log_every_n_steps=1,
        #logger=mlflow_logger
        #limit_train_batches=0.01,
        #limit_val_batches=0.02
      )

  #fit
  logger.info("Run trainer to fit the model")
  trainer.fit(model, dm)
    
  logger.info("test the best model")
  trainer.test(ckpt_path="best", datamodule=dm)
  # trainer.test()
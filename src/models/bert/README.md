Implementation of BERT for Text Classification
==============================

### To use the code ###
1) Clone the repo
2) Setup a virtual env by using environment.yaml
3) Change directory to src/models/bert
4) Run the train_model.py file

* Modify model name and data paths according to your choice

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── arguments.py       <- Hyperparameters and arguments for model.
    │
    ├── config.py          <- Data paths and model name.
    │
    ├── dataloader.py      <- Reads and loads the data.
    │
    ├── dataset.py         <- Tokenize the text and converts data frame to a dataset.
    │
    ├── logger.py          <- Logger to track the run.
    │
    ├── model.py           <- Script for text classifier.
    │
    └── train_model.py     <- Script to train models.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

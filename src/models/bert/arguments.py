#imports
import argparse
import torch

#arguments
def get_args():
    parser = argparse.ArgumentParser(description = "handle arguments")
    

    parser.add_argument("-max_length", type = int, help = "Maximum Token Length of BERT", 
        default = 140
    )

    parser.add_argument("-learning_rate", type = float, help = "Learning Rate of Optimizer", 
        default = 2e-5
    )

    parser.add_argument("-dropout", type = float, help = "Dropout probability of linear classifier", 
        default =  0.3
    )

    parser.add_argument("-weight_decay", type = float, help = "Weight Decay of Optimizer", 
        default = 2e-5  #1e-6
    )

    parser.add_argument("-epsilon", type = float, help = "Epsilon of Optimizer", 
        default = 4e-5   #1e-6
    )

    parser.add_argument("-batch_size", type = int, help = "Batch size",
        default = 50    #if min(1, torch.cuda.device_count()) else 25
    )

    parser.add_argument("-max_epochs", type = int, help = "Max number of epochs neural net",
        default = 3  #10
    )

    parser.add_argument("-random_seed", type = int, help = "Random seed for seedification",
        default = 42
    )

    parser.add_argument("-num_workers", type = int, help = "Number of workers",
        default = 4
    )

    parser.add_argument("-gpus", type = int, help = "Number of GPUs", 
        default = torch.cuda.device_count()
    )

    args = parser.parse_args()

    return args

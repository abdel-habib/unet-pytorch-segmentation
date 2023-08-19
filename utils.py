import os
import random
import numpy as np
import torch

def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins, elapsed_secs = divmod(elapsed_time, 60)
    return elapsed_mins, elapsed_secs

def check_data_empty(data_x, data_y, data_name):
    if len(data_x) == 0 or len(data_y) == 0:
        raise ValueError(f'Missing {data_name} data. Please check if the given paths are correct.')


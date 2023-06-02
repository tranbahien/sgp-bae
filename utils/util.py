import numpy as np
import torch
import pickle
import glob
import json

from itertools import repeat
from collections import OrderedDict
from pathlib import Path


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def set_seed(seed=0):
    """Set seed for reproducibility purpose."""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def get_all_files(pattern):
    """Get all files located in a directory given a pattern."""
    file_paths = glob.glob(pattern)

    if len(file_paths) < 1:
        return None
    else:
        return file_paths


def save_pickle(data, file_path):
    """Wrapper for saving data to a pickle file.

    Args:
        data: a dictionary containing the data needs to be saved.
        file_path: string, path to the output file.
    """
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    """Wrapper for loading data from a pickle file.

    Args:
        file_path: string, path to the pickle file.

    Returns:
        A dictionary containing the loaded data.
    """
    with open(file_path, 'rb') as handle:
        data = pickle.load(handle)
    return data


def ensure_dir(dirname):
    """Check whether the given directory was created; if not, create a new one.

    Args:
        dirname: string, path to the directory.
    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)
        
def read_json(file_path):
    """Wrapper for reading a json file.

    Args:
        file_path: string, path to the json file.

    Returns:
        A dictionary containing the loaded data.
    """
    file_path = Path(file_path)
    with file_path.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)
    

def to_one_hot(y, num_classes=10):
    """Convert labels to one-hot vectors.

    Args:
        y: numpy array, shape [num_classes], the true labels.

    Returns:
        one_hot: numpy array, size (?, num_classes), 
            array containing the one-hot encoding of the true classes.
    """
    if isinstance(y, torch.Tensor):
        one_hot = torch.zeros((y.shape[0], num_classes), dtype=torch.float32)
        one_hot[torch.arange(y.shape[0]), y] = 1.0
    else:
        one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float)
        one_hot[np.arange(y.shape[0]), y] = 1.0

    return one_hot
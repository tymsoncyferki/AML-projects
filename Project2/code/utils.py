import numpy as np
import pandas as pd

def custom_score(y_true: np.ndarray, y_prob: np.ndarray, num_features: int, reference_size: int = 5000, top_k_reference: int = 1000):
    """
    Calculates score: (10 * TP * scaling) - (200 * n_variables), takes into account the proportions of the data.

    Args:
        y_true (np.array): True binary labels (0 or 1) indicating actual energy usage status
        y_prob (np.array): Predicted probabilities
        num_features (int): Number of features used by the model
        reference_size (int): Size of the full dataset the top-k selection is based on (default=5000)
        top_k_reference (int): Number of households to target in the full dataset (default=1000)

    Returns:
        float: custom score
    """
    n = len(y_true)
    top_k = int(top_k_reference * n / reference_size)
    top_idx = np.argsort(y_prob)[-top_k:]
    tp = y_true[top_idx].sum()

    scaling_factor = reference_size / n
    score = 10 * tp * scaling_factor - 200 * num_features
    return score

def load_data(filename: str, path: str = "../data/") -> pd.DataFrame:
    """
    Loads data

    Args:
        filename (str): name of the file to load e.g., 'x_train.txt'
        path (str): path to the folder containing the data, default is '../data/'

    Returns:
        pd.DataFrame: loaded data
    """
    return pd.read_table(path + filename, sep=" ", header=None)
    
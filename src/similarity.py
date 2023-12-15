import numpy as np
import numpy.typing as npt

def cosine_similarity(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """
    Calculate the cosine similarity of 2 numpy arrays

    Return values range from -1 to 1, where 1 = equal, and -1 = opposite
    """
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

def euclidian_distance(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """
    Calculate the euclidian distance between 2 vectors, represented as 2 numpy arrays

    Return values ranging from 0, to the length of the longest vector
    """
    return np.linalg.norm(a-b)
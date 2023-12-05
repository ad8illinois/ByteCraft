import numpy as np
import numpy.typing as npt

def vector_to_prob_dist(vector) -> npt.ArrayLike:
    """
    Convert a term-frequency vector to a probability distribution.

    :param tf_vector: A numpy vector respresenting number of occurences for each word
    :param pseudo_counts: A numpy vector respresenting number of occurences for each word
    """
    doc_length = np.sum(vector)
    prob_dist = vector / doc_length

    # TODO: apply smoothing to this function, potentially using the corpus lm as a reference for adding pseudo-counts?

    return prob_dist

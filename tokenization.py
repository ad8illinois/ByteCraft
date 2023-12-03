from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.cluster import AgglomerativeClustering
import string
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm 
import numpy.typing as npt

"""
NOTE: Your first time running this script, you will have to install nltk, and run the below in a python interpreter:

import nltk
nltk.download()

This will open a window giving you corpuses to download. I just downloaded the 'popular' option, to the default location.
You only have to do this once for every machine.
"""

def tokenize_file(filepath: str):
    """
    Use NLTK to tokenize the given text file.
    Makes all tokens lowercase, and performs basic stop-word removal.

    Args:
        filepath (str): Absolute or relative filepath to the input text file
    Returns:
        List of strings, each representing a token extracted from the text.
    """
    with open(filepath, 'r') as file:
        text = file.read()

        # Make everything lowercase, to prevent duplicates that only differ in casing 
        text = text.lower()

        # Set of english stop words and punctuation marks
        stop = set(stopwords.words('english') + list(string.punctuation))

        # Perform tokenization, removing stop words
        tokens = [i for i in nltk.word_tokenize(text) if i not in stop]

        """
        TODO: some ideas for reducing the number of tokens, without hurting accuracy too much
            - stemming
            - convert numbers into placeholders
        """
        return tokens


#  # I wrote this code, but I don't think we actually need it. Because we're using dictionaries instead of vectors for term-frequency, we already know what the tokens are.
#  # We don't need a lookup dictionary. 
#  # 
#  # In the future, we may want to go back from dicts to vectors to make computation more efficient, then, it'll be useful to have a single array which defines what each index means.
# 
# 
# def create_dictionary(filepaths: List[str]) -> List[str]: 
#     """
#     Read all the files in given, and create a dictionary which encapsulates all of the tokens present across all files.

#     Args:
#         filepaths (List[str]): List of filepaths, all the input files in your corpus

#     Returns:
#         arr of strings, representing all the tokens found in the text
#     """
#     tokens_set = set()
#     for filepath in filepaths:
#         print(f'Reading file for tokens: {filepath}')
#         tokens = tokenize_file(filepath)
#         print(f'Found {len(tokens)} tokens in file')
#         unique_tokens = set(tokens)
#         print(f'Found {len(unique_tokens)} unique tokens in file')
#         tokens_set = tokens_set | unique_tokens

#     return list(tokens_set)  # NOTE to self, as dictionary size grows, we should think about using numpy arrays for dictionaries. Numpy arrays are generally faster and take less memory than lists.


def create_term_frequency_dict(filepath: str) -> List[int]:
    """
    Read a file, and convert it into a term-frequency vector based on the given dictionary
    """
    tokens = tokenize_file(filepath)
    freq_dist = FreqDist(tokens)
    tf_dict = dict(freq_dist)

    return tf_dict


def get_tf_vector(tf_reverse_index, filepath: str) -> npt.ArrayLike:
    """
    Create a tf-vector for a given filepath, using the given reverse index.
    Guaranteed to give comparable vectors between filepaths AS LONG AS the same reverse index is used.
    """
    num_tokens = len(tf_reverse_index)
    vector = np.zeros((num_tokens))

    for i, token in enumerate(tf_reverse_index):
        if filepath in tf_reverse_index[token]:
            vector[i] = tf_reverse_index[token][filepath]
        else:
            vector[i] = 0
    
    return vector

def cosine_similarity(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.ArrayLike:
    """
    Calculate the cosine similarity of 2 numpy arrays

    Return values range from -1 to 1, where 1 = equal, and -1 = opposite
    """
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

# K-means clustering traditionally requires a euclidean or cosine distance between vectors and not a similarity
# matrix. Popular python libraries provide k-means implementations that expect standard distance metrics conforming
# to euclidean/cosine distances.

# Agglomerative Clustering is much better suited for using a similarity function out-of-the-box
# Although the implementation denotes the use of a distance matrix instead of a similarity matrix,
# they are simply the inverse of each other.
# So we can precompute a distance matrix as 1-(similairty matrix) before passing it into the clustering function
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

def agglomerative_clustering(similarity_matrix, n_clusters):
    # precompute a distance matrix as inverse of similarity_matrix before passing it into the clustering function
    distance_matrix = 1 - similarity_matrix
    # We can switch back to the default linkage "ward" based on performance
    clustering_model = AgglomerativeClustering(metric="precomputed", linkage="average", n_clusters=n_clusters).fit(
        distance_matrix)
    # Cluster labels [0,1]
    return clustering_model.labels_


if __name__ == '__main__':
    base_dir = './testdata/numpy/issues'
    # base_dir = './testdata/wikipedia'
    # base_dir = './testdata/dummy'

    output_base_dir = './output'
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # Collect all filepaths in this dir
    filepaths = []
    for root, _, files in os.walk(base_dir, topdown=False):
        for name in files:
            if name.endswith('.txt'):
                filepaths.append(os.path.join(root, name))

    # # See the comment on create_dictionary
    # 
    # print('Creating dictionary...')
    # dictionary = create_dictionary(filepaths)
    # print(f'Parsed dictionary of {len(dictionary)} tokens')

    # Stores term frequencies across all files, indexed by token, then filepath, then count
    tf_reverse_index = {
        # Example record in this reverse index:
        # 
        # 'token': {
        #    './testdata/dummy/1.txt': 1,
        # }
    }
    for filepath in filepaths:
        print(f'Counting tokens in file: {filepath}')
        tf_dict = create_term_frequency_dict(filepath)

        for token in tf_dict:
            count = tf_dict[token]
            if token not in tf_reverse_index:
                tf_reverse_index[token] = {}
            tf_reverse_index[token][filepath] = count

    # Write the reverse index to a text file
    print('Writing reverse index to file.')
    with open(os.path.join(output_base_dir, 'tf_reverse_index.txt'), 'w') as output_file:
        for token in tf_reverse_index:
            output_file.write(f'{token} - {tf_reverse_index[token]}\n')

    # Go back through each file, and calculate tf-vectors
    # 
    # Because we're using the same tf_reverse_index each time, these vectors will have the same dimensions, 
    # and therefore, can be compared using standard vector similarity algorithms. 
    tf_vectors = {}
    for filepath in filepaths:
        print(f"Calculating tf-vector for file: {filepath}")
        tf_vector = get_tf_vector(tf_reverse_index, filepath)
        tf_vectors[filepath] = tf_vector
    
    """
    TODO: Apply TF-transformation and IDF to each TF-vector.
    """

    # Write each tf-vector to a file. For debugging and sanity checks.
    for filepath in tf_vectors:
        relative_filepath = filepath.replace('./', '')
        output_filepath = os.path.join(output_base_dir, relative_filepath)
        base_dir = os.path.dirname(output_filepath)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        np.savetxt(output_filepath, tf_vectors[filepath], fmt='%4f', delimiter=', ', newline='\n')

    # Calculate a similarity matrix
    print('Calculating similarity matrix')
    sim_matrix = np.zeros((len(filepaths), len(filepaths)))
    for a, filepath_a in enumerate(filepaths):
        for b, filepath_b in enumerate(filepaths):
            vec_a = tf_vectors[filepath_a]
            vec_b = tf_vectors[filepath_b]
            sim_matrix[a][b] = cosine_similarity(vec_a, vec_b)

    np.savetxt(os.path.join(output_base_dir, './sim_matrix.txt'), sim_matrix, fmt='%4f', delimiter=', ', newline='\n', header=', '.join(filepaths))

    # Agglomerative clustering
    print('Performing agglomerative clustering with 4 clusters')
    clustering = agglomerative_clustering(similarity_matrix=sim_matrix, n_clusters=4) # There are 4 authors
    print('Clustering Results', clustering)

    for i, filepath in enumerate(filepaths):
        print(f"File: {filepath}  Cluster: {clustering[i]}")

    # print(sim_matrix)

    # TODO / Next Steps
    #  - Given a doc, use the reverse index to calculate a tf-idf vector for that document
    #      - Implement a similarity function between 2 tf-idf vectors
    #            - Using this similarity function, implement KNN

    #  - Given a list of docs in a topic, calculate a unigram LM for that Topic
    #       -  Implement smoothing on the LM, so we get on-zero probabilities for every word in the topic
    #       -  Implement the Naive Bayes scoring function, given a topic LM and a doc, what's the log likelihood of generating that doc with that LM, usign bayes rule to account for the distribution of topics themselves
    #       -  Use the Naive bayes to classify any new doc into one of the pre-defined topics



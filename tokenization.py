from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string
import os
import numpy as np
import numpy.typing as npt
from ml_model_definitions import agglomerative_clustering
from sklearn.feature_extraction.text import TfidfTransformer

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
    Smooth the vector by adding pseudo counts (1) to each word.
    Guaranteed to give comparable vectors between filepaths AS LONG AS the same reverse index is used.
    """
    num_tokens = len(tf_reverse_index)
    vector = np.zeros((num_tokens))

    for i, token in enumerate(tf_reverse_index):
        if filepath in tf_reverse_index[token]:
            vector[i] = tf_reverse_index[token][filepath] + 1
        else:
            vector[i] = 1
    
    return vector

def tf_vector_to_unigram_lm(tf_vector) -> npt.ArrayLike:
    """
    Convert a term-frequency vector to a Unigram LM probability distribution, using Dirichlet smoothing

    :param tf_vector: A numpy vector respresenting number of occurences for each word
    :param pseudo_counts: A numpy vector respresenting number of occurences for each word
    """
    doc_length = np.sum(tf_vector)
    lm = tf_vector / doc_length

    # TODO: apply smoothing to this function, potentially using the corpus lm as a reference for adding pseudo-counts?

    return lm

def calc_topic_unigram_lm(tf_reverse_index, filepaths: List[str]) -> npt.ArrayLike:
    """
    Define a topic as a collection of documents, and calculate the Max-Likelihood Unigram LM for that topic.

    Internally uses tf_vector_to_unigram_lm, so any smoothing parameters used by that function can also apply
    """
    topic_tf_vector = np.zeros((len(tf_reverse_index)))
    for filepath in filepaths:
        tf_vector = get_tf_vector(tf_reverse_index, filepath)
        topic_tf_vector = topic_tf_vector + tf_vector
    
    lm = tf_vector_to_unigram_lm(topic_tf_vector)
    return lm

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

def word_vec_to_file(word_dict, vector, filepath):
    """
    Write a word vector to a filepath, in a human readable format.
    Includes the words themselves.
    """
    with open(filepath, 'w') as file:
        for i, token in enumerate(word_dict):
            file.write(f'{token}: {vector[i]}\n')

def learn_topics(topic_documents: dict[str, List[str]]):
    """
    Learn Unigram Topic LMs from the given data-set.

    Input should be a dict mapping from topic names, to a list of files which belong to that topic
    Example:
    {
        'animals': ['./data/bird.txt', './data/dog.txt'],
        'cities': ['./data/chicago.txt', './data/newyork.txt']
    }

    Returns: ['bird', 'cat', 'weather', ...], {
            '_corpus': [0.5, 0.0001, 0.0002......],
            'animals': [0.5, 0.0001, 0.0002......],
            'cities': [0.5, 0.0001, 0.0002......]
    }
    """
    filepaths = []
    for topic in topic_documents:
        for filepath in topic_documents[topic]:
            filepaths.append(filepath)

    # Stores term frequencies across all files, indexed by token, then filepath, then count.
    # This dict is also used frequently as the token dictionary, because all tokens must exist in this data structure.
    print('Creating tf reverse index...')
    tf_reverse_index = {
        # Example record in this reverse index:
        # 
        # 'token': {
        #    './testdata/dummy/1.txt': 1,
        # }
    }
    for filepath in filepaths:
        print(f'  Counting tokens in file: {filepath}')
        tf_dict = create_term_frequency_dict(filepath)

        for token in tf_dict:
            count = tf_dict[token]
            if token not in tf_reverse_index:
                tf_reverse_index[token] = {}
            tf_reverse_index[token][filepath] = count
    
    tokens = [token for token in tf_reverse_index]

    # # Write the reverse index to a text file
    # with open(os.path.join(output_base_dir, 'tf_reverse_index.txt'), 'w') as output_file:
    #     for token in tf_reverse_index:
    #         output_file.write(f'{token} - {tf_reverse_index[token]}\n')


    # Go back through each file, and calculate tf-vectors
    # 
    # Because we're using the same tf_reverse_index each time, these vectors will have the same dimensions, 
    # and therefore, can be compared using standard vector similarity algorithms. 
    print('Creating tf-vectors for all files...')
    doc_tf_vectors = {}

    for filepath in filepaths:
        print(f"  Calculating tf-vector for file: {filepath}")
        topic_tf_vector = get_tf_vector(tf_reverse_index, filepath)
        doc_tf_vectors[filepath] = topic_tf_vector

    # # Write each tf-vector to a file. For debugging and sanity checks.
    # for filepath in tf_vectors:
    #     relative_filepath = filepath.replace('./', '').replace('.txt', '_tf.txt')
    #     output_filepath = os.path.join(output_base_dir, relative_filepath)
    #     base_dir = os.path.dirname(output_filepath)
    #     if not os.path.exists(base_dir):
    #         os.makedirs(base_dir)

    #     word_vec_to_file(tf_reverse_index, tf_vectors[filepath], output_filepath)

    # word_vec_to_file(tf_reverse_index, corpus_tf_vector, './output/corpus_tf_vector.txt')


    print('Creating unigram LMs for all topics')
    topic_lms = {}
    for topic in topic_documents:
        print(f"  Calculating unigram LM for topic: {topic}")
        topic_tf_vector = np.zeros((len(tf_reverse_index))) 
        for filepath in topic_documents[topic]:
            topic_tf_vector = topic_tf_vector + doc_tf_vectors[filepath]
        topic_lms[topic] = tf_vector_to_unigram_lm(topic_tf_vector)


    # Write each unigram_lm to a file. For debugging and sanity checks.
    # for filepath in doc_tf_vectors:
    #     relative_filepath = filepath.replace('./', '').replace('.txt', '_lm.txt')
    #     output_filepath = os.path.join(output_base_dir, relative_filepath)
    #     base_dir = os.path.dirname(output_filepath)
    #     if not os.path.exists(base_dir):
    #         os.makedirs(base_dir)

    #     np.savetxt(output_filepath, unigram_lms[filepath], fmt='%4f', delimiter=', ', newline='\n')
    #     word_vec_to_file(tf_reverse_index, unigram_lms[filepath], output_filepath)

    return tokens, topic_lms


if __name__ == '__main__':
    # base_dir = './testdata/numpy/issues'
    base_dir = './testdata/wikipedia'
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

    # Stores term frequencies across all files, indexed by token, then filepath, then count.
    # This dict is also used frequently as the token dictionary, because all tokens must exist in this data structure.
    print()
    print('Creating tf reverse index...')
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
    with open(os.path.join(output_base_dir, 'tf_reverse_index.txt'), 'w') as output_file:
        for token in tf_reverse_index:
            output_file.write(f'{token} - {tf_reverse_index[token]}\n')


    # Go back through each file, and calculate tf-vectors
    # 
    # Because we're using the same tf_reverse_index each time, these vectors will have the same dimensions, 
    # and therefore, can be compared using standard vector similarity algorithms. 
    print()
    print('Creating tf-vectors for all files, and for corpus...')
    corpus_tf_vector = np.zeros((len(tf_reverse_index)))
    tf_vectors = {}

    for filepath in filepaths:
        print(f"Calculating tf-vector for file: {filepath}")
        tf_vector = get_tf_vector(tf_reverse_index, filepath)
        tf_vectors[filepath] = tf_vector

        corpus_tf_vector = corpus_tf_vector + tf_vector

    # Write each tf-vector to a file. For debugging and sanity checks.
    for filepath in tf_vectors:
        relative_filepath = filepath.replace('./', '').replace('.txt', '_tf.txt')
        output_filepath = os.path.join(output_base_dir, relative_filepath)
        base_dir = os.path.dirname(output_filepath)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        word_vec_to_file(tf_reverse_index, tf_vectors[filepath], output_filepath)

    word_vec_to_file(tf_reverse_index, corpus_tf_vector, './output/corpus_tf_vector.txt')

    print()
    print('Creating unigram LMs for all files, and for corpus...')
    unigram_lms = {}
    for filepath in filepaths:
        print(f"Calculating unigram LM for file: {filepath}")
        tf_vector = tf_vectors[filepath]
        unigram_lm = tf_vector_to_unigram_lm(tf_vector)
        unigram_lms[filepath] = unigram_lm

    corpus_unigram_lm = tf_vector_to_unigram_lm(corpus_tf_vector)
    word_vec_to_file(tf_reverse_index, corpus_unigram_lm, './output/corpus_unigram_lm.txt')

    # Write each unigram_lm to a file. For debugging and sanity checks.
    for filepath in tf_vectors:
        relative_filepath = filepath.replace('./', '').replace('.txt', '_lm.txt')
        output_filepath = os.path.join(output_base_dir, relative_filepath)
        base_dir = os.path.dirname(output_filepath)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        np.savetxt(output_filepath, unigram_lms[filepath], fmt='%4f', delimiter=', ', newline='\n')
        word_vec_to_file(tf_reverse_index, unigram_lms[filepath], output_filepath)
    
    """
    TODO: Apply TF-transformation and IDF to each TF-vector.
    """

    print()
    print('Calculating similarity matrix')

    # Calculate a similarity matrix
    distance_matrix = np.zeros((len(filepaths), len(filepaths)))
    for a, filepath_a in enumerate(filepaths):
        for b, filepath_b in enumerate(filepaths):
            vec_a = tf_vectors[filepath_a]
            vec_b = tf_vectors[filepath_b]

            distance_matrix[a][b] =  1 - cosine_similarity(vec_a, vec_b)
            # distance_matrix[a][b] = euclidian_distance(vec_a, vec_b)

    np.savetxt(os.path.join(output_base_dir, './distance_matrix.txt'), distance_matrix, fmt='%4f', delimiter=', ', newline='\n', header=', '.join(filepaths))

    # Agglomerative clustering
    print('Performing agglomerative clustering with 4 clusters')
    clustering = agglomerative_clustering(distance_matrix=distance_matrix, n_clusters=3)
    print('')
    print('Clustering Results:')

    for i, filepath in enumerate(filepaths):
        print(f"File: {filepath}  Cluster: {clustering[i]}")


    #  - Given a list of docs in a topic, calculate a unigram LM for that Topic
    #       -  Implement smoothing on the LM, so we get on-zero probabilities for every word in the topic
    #       -  Implement the Naive Bayes scoring function, given a topic LM and a doc, what's the log likelihood of generating that doc with that LM, usign bayes rule to account for the distribution of topics themselves
    #       -  Use the Naive bayes to classify any new doc into one of the pre-defined topics



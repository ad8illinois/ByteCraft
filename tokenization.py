import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string
# from sklearn.feature_extraction.text import TfidfTransformer

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

def create_term_frequency_dict(filepath: str):
    """
    Read a file, and convert it into a term-frequency vector based on the given dictionary
    """
    tokens = tokenize_file(filepath)
    freq_dist = FreqDist(tokens)
    tf_dict = dict(freq_dist)

    return tf_dict


# if __name__ == '__main__':
#     """
#     TODO: Apply TF-transformation and IDF to each TF-vector.
#     """

#     print()
#     print('Calculating similarity matrix')

#     # Calculate a similarity matrix
#     distance_matrix = np.zeros((len(filepaths), len(filepaths)))
#     for a, filepath_a in enumerate(filepaths):
#         for b, filepath_b in enumerate(filepaths):
#             vec_a = tf_vectors[filepath_a]
#             vec_b = tf_vectors[filepath_b]

#             # distance_matrix[a][b] =  1 - cosine_similarity(vec_a, vec_b)
#             distance_matrix[a][b] = euclidian_distance(vec_a, vec_b)

#     np.savetxt(os.path.join(output_base_dir, './distance_matrix.txt'), distance_matrix, fmt='%4f', delimiter=', ', newline='\n', header=', '.join(filepaths))

#     # Agglomerative clustering
#     print('Performing agglomerative clustering with 4 clusters')
#     clustering = agglomerative_clustering(distance_matrix=distance_matrix, n_clusters=3)
#     print('')
#     print('Clustering Results:')

#     for i, filepath in enumerate(filepaths):
#         print(f"File: {filepath}  Cluster: {clustering[i]}")

    wikipedia_topics = { 
        'animals': ['./testdata/wikipedia/bird.txt', './testdata/wikipedia/cat.txt', './testdata/wikipedia/dog.txt', './testdata/wikipedia/fish.txt'], 
        'places': ['./testdata/wikipedia/chicago.txt', './testdata/wikipedia/champaign.txt', './testdata/wikipedia/uiuc.txt']
    }

    dummy_topics = {
        'happy': ['./testdata/dummy/1.txt', './testdata/dummy/3.txt'],
        'sad': ['./testdata/dummy/2.txt']
    }

    tokens, lm = learn_topics(dummy_topics)
    # print(tokens, lm)

    X = dummy_topics.values()
    # y = dummy_topics.keys()
    X = ['./testdata/dummy/1.txt'], ['./testdata/dummy/3.txt'], ['./testdata/dummy/2.txt']
    y = ['happy', 'happy', 'sad']
    print(list(X))
    print(list(y))

    # KNN Classification
    print('Performing KNN classification with k=2...')
    knn = knn_classification(2, X, y)
    print('')
    print('Classification Results:')
    print(knn)


#     #  - Given a list of docs in a topic, calculate a unigram LM for that Topic
#     #       -  Implement smoothing on the LM, so we get on-zero probabilities for every word in the topic
#     #       -  Implement the Naive Bayes scoring function, given a topic LM and a doc, what's the log likelihood of generating that doc with that LM, usign bayes rule to account for the distribution of topics themselves
#     #       -  Use the Naive bayes to classify any new doc into one of the pre-defined topics



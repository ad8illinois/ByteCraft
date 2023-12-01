from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string
import os

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
    with open (filepath, 'r') as file:
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

if __name__ == '__main__':
    base_dir = './testdata/numpy/issues'
    # base_dir = './testdata/wikipedia'
    # base_dir = './testdata/dummy'

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
    with open('./tf_reverse_index.txt', 'w') as output_file:
        for token in tf_reverse_index:
            output_file.write(f'{token} - {tf_reverse_index[token]}\n')


    # TODO / Next Steps
    #  - Given a doc, use the reverse index to calculate a tf-idf vector for that document
    #      - Implement a similarity function between 2 tf-idf vectors
    #            - Using this similarity function, implement KNN

    #  - Given a list of docs in a topic, calculate a unigram LM for that Topic
    #       -  Implement smoothing on the LM, so we get on-zero probabilities for every word in the topic
    #       -  Implement the Naive Bayes scoring function, given a topic LM and a doc, what's the log likelihood of generating that doc with that LM, usign bayes rule to account for the distribution of topics themselves
    #       -  Use the Naive bayes to classify any new doc into one of the pre-defined topics

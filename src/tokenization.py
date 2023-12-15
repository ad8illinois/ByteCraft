import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string

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

def create_tf_dict(filepath: str):
    """
    Read a file, and convert it into a term-frequency vector based on the given dictionary
    """
    tokens = tokenize_file(filepath)
    freq_dist = FreqDist(tokens)
    tf_dict = dict(freq_dist)

    return tf_dict

def count_total_tokens_in_doc(filepath: str):
    """
    Read a file, and compute the total number of terms in a given document
    """
    tokens = tokenize_file(filepath)
    return len(tokens)

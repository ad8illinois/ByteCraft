import nltk
from nltk.corpus import stopwords
import string
import os

"""
NOTE: Your first time running this script, you will have to install nltk, and run the below in a python interpreter:

import nltk
nltk.download()

This will open a window giving you corpuses to download. I just downloaded the 'popular' option, to the default location.
You only have to do this once for every machine.
"""


base_dir = './testdata/numpy/issues'
contributor_filenames = os.listdir(base_dir)

for contributor_filename in contributor_filenames:
    contributor = contributor_filename
    contributor_dir = os.path.join(base_dir, contributor_filename)

    issue_filenames = os.listdir(contributor_dir)
    for issue_filename in issue_filenames:
        issue_filepath = os.path.join(contributor_dir, issue_filename)
        issue_key = issue_filename.replace('.txt', '')

        print(f'Tokenizing issue: {issue_key} from contributor: {contributor}')
        with open (issue_filepath, 'r') as issue_file:
            text = issue_file.read()

            # Make everything lowercase, to prevent duplicates that only differ in casing 
            text = text.lower()

            # Set of english stop words and punctuation marks
            stop = set(stopwords.words('english') + list(string.punctuation))

            # Perform tokenization, removing stop words
            tokens = [i for i in nltk.word_tokenize(text) if i not in stop]
            print(tokens)


# TODO / Next Steps
# 
#  - Merge all the tokenization arrays into a single vector, representing all the tokens found in the entire corpus
#  - Write a function for converting a token array into a TF vector
#       - NLTK's FreqDist class may be useful here: https://www.nltk.org/api/nltk.probability.FreqDist.html
# 

# Potential function to convert tokens into TF vectors

from nltk.probability import FreqDist

def tokens_to_tf_vector(tokens):
    # Calculate term frequencies using NLTK's FreqDist
    freq_dist = FreqDist(tokens)
    
    # Convert FreqDist to a dictionary
    tf_dict = dict(freq_dist)
    
    return tf_dict

tf_vector = tokens_to_tf_vector(tokens)
print("TF Vector:")
print(tf_vector)
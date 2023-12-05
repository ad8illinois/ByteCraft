from tokenization import create_term_frequency_dict
import numpy as np

class InvertedIndex:
    def __init__(self):
        self.term_counts = {
            # 'term': {
            #    './filepath.txt': 1,
            # }
        }
    
    def add_document(self, filepath: str):
        """
        Add a document to the inverted index, tokenizes the document, and adds counts to the index.

        NOTE: this function is idempotent, there is no harm in indexing a document multiple times.
        """
        print(f'Adding document to inverted index: {filepath}')
        doc_term_counts = create_term_frequency_dict(filepath)
        for term in doc_term_counts:
            if term not in self.term_counts:
                self.term_counts[term] = {}
            
            count = doc_term_counts[term]
            self.term_counts[term][filepath] = count
            
        self.words = [word for word in self.term_counts]
    
    def get_terms(self):
        """
        Return all terms as a list / vector, in the same order as get_tf_vector.
        """
        return [word for word in self.term_counts]

    def get_tf_vector(self, filepath: str, pseudo_counts = 0):
        """
        Given a filepath, returns a term-frequency vector corresponding to all the terms in that filepath
        File should have been indexed using add_document(). If the file was not indexed, you will receive all 0's.

        Optionally smooth the vector by adding pseudo counts to each word.
        """
        num_terms = len(self.term_counts)
        vector = np.zeros((num_terms))

        for i, term in enumerate(self.term_counts):
            if filepath in self.term_counts[term]:
                vector[i] = self.term_counts[term][filepath] + pseudo_counts
            else:
                vector[i] = pseudo_counts
        
        return vector
    
    def export_to_file(self, filepath: str):
        """
        Write the inverted index to a file, for debugging or demoing
        """
        with open(filepath, 'w') as output_file:
            for token in self.term_counts:
                output_file.write(f'{token} - {self.term_counts[token]}\n')

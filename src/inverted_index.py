from tokenization import create_tf_dict
import numpy as np
import json

class InvertedIndex:
    def __init__(self):
        self.term_counts = {
            # 'term': {
            #    './filepath.txt': 1,
            # }
        }
        self.num_docs = 0
        self.total_terms_in_doc = {}
        self.file_paths = []

    def add_document(self, filepath: str):
        """
        Add a document to the inverted index, tokenizes the document, and adds counts to the index.

        NOTE: this function is idempotent, there is no harm in indexing a document multiple times.
        """
        print(f'Adding document to inverted index: {filepath}')
        self.file_paths.append(filepath)
        doc_term_counts = create_tf_dict(filepath)
        for term in doc_term_counts:
            if term not in self.term_counts:
                self.term_counts[term] = {}
            
            count = doc_term_counts[term]
            self.term_counts[term][filepath] = count
            
        self.num_docs = self.num_docs+1
    
    def get_terms(self):
        """
        Return all terms as a list / vector, in the same order as get_tf_vector.
        """
        return [word for word in self.term_counts]

    def tf_dict_to_vector(self, tf_dict, pseudo_counts = 0):
        num_terms = len(self.term_counts)
        vector = np.zeros((num_terms))

        for i, term in enumerate(self.term_counts):
            if term in tf_dict:
                vector[i] = tf_dict[term] + pseudo_counts
            else:
                vector[i] = pseudo_counts
        return vector


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
                vector[i] = self.term_counts[term][filepath]+ pseudo_counts
            else:
                vector[i] = pseudo_counts
        
        return vector
    
    def export_to_file(self, filepath: str):
        """
        Write the inverted index to a file, for debugging or demoing
        """
        with open(filepath, 'w') as output_file:
            output_file.write(f'Num docs:{self.num_docs}\n')
            for token in self.term_counts:
                term_counts_json = json.dumps(self.term_counts[token])
                output_file.write(f'{token} - {term_counts_json}\n')

    def load_from_file(self, filepath: str):
        """
        Initialize this inverted index with the contents of the file
        """

        self.term_counts = {}
        with open(filepath, 'r') as file:
            for line in file:
                if "Num docs" in line:
                    self.num_docs = int(line.split(':')[1])
                    continue
                token = line.split(' - ')[0]
                term_counts_json = line.split(' - ')[1]
                term_counts = json.loads(term_counts_json)
                self.term_counts[token] = term_counts

    def compute_tf_idf_vector(self):

        unique_words_in_corpus = self.get_terms()
        idf = {}
        tf_np = np.zeros((self.num_docs, len(unique_words_in_corpus)))

        for i, word in enumerate(unique_words_in_corpus):
            idf[word] = np.log10(self.num_docs/len(self.term_counts[word])+1)

        for doc_id, file in enumerate(self.file_paths):
            for i, word in enumerate(unique_words_in_corpus):
                tf_np[doc_id, i] = self.term_counts[word].get(file, 0)/self.total_terms_in_doc[file]

        tf_idf_vector = tf_np * np.array(list(idf.values()))

        return tf_idf_vector

    def apply_tf_transformation(self, tf_vector):
        """
        Applies bm25 tf-transformation, with k=5
        """
        k=5
        return (tf_vector * (k+1)) / (tf_vector + k)
    
    def apply_idf(self, tf_vector):
        """
        Applies idf normalization to the given tf-vector
        """
        doc_freq_vector = np.zeros(len(self.term_counts))
        for i, term in enumerate(self.term_counts):
            doc_freq_vector[i] = len(self.term_counts[term])
        
        idf_vector = np.log((self.num_docs + 1) / doc_freq_vector)
        return tf_vector * idf_vector


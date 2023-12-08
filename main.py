import click
import os
import numpy as np
from inverted_index import InvertedIndex
from prob_dist import vector_to_prob_dist
from util import term_vec_to_file
from ml_model_definitions import knn_classification

@click.group()
def cli():
    pass

@click.command()
@click.option('--project-url', help='Full url to the project on github')
@click.option('--api-token', help='Github API token ')
@click.option('--dir', help='Download location')
def download(project_url, api_token, dir):
    """
    TODO: Download all the issues from a github repo, and save them into a local folder
    """
    print('TODO: unimplemented')

@click.command()
@click.option('--input-dir', help='Folder with training data')
@click.option('--output-dir', default='./output', help='Folder to put LMs into')
def learn(input_dir, output_dir):
    """
    Do topic modeling on a set of input text files with known classifications.

    Estimates a ML Unigram LM for each topic given, and writes the LMs to an output directory.
    """
    # topic_documents = {
    #     'animals': [
    #         './testdata/wikipedia/bird.txt',
    #         './testdata/wikipedia/cat.txt',
    #         './testdata/wikipedia/dog.txt',
    #         './testdata/wikipedia/fish.txt',
    #     ],
    #     'places': [
    #         './testdata/wikipedia/champaign.txt',
    #         './testdata/wikipedia/chicago.txt',
    #         './testdata/wikipedia/uiuc.txt',
    #     ],
    #     'corpus': {
    #         './testdata/wikipedia/bird.txt',
    #         './testdata/wikipedia/cat.txt',
    #         './testdata/wikipedia/dog.txt',
    #         './testdata/wikipedia/fish.txt',
    #         './testdata/wikipedia/champaign.txt',
    #         './testdata/wikipedia/chicago.txt',
    #         './testdata/wikipedia/uiuc.txt',
    #     }
    # }

    topic_documents = {
        'happy': [
            './testdata/dummy/1.txt',
            './testdata/dummy/3.txt',
            './testdata/dummy/4.txt',
            './testdata/dummy/6.txt',
        ],
        'sad': [
            './testdata/dummy/2.txt',
            './testdata/dummy/5.txt',
        ],
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'documents')):
        os.makedirs(os.path.join(output_dir, 'documents'))
    if not os.path.exists(os.path.join(output_dir, 'categories')):
        os.makedirs(os.path.join(output_dir, 'categories'))

    filepaths = []
    for topic in topic_documents:
        for filepath in topic_documents[topic]:
            filepaths.append(filepath)
    
    # Construct inverted index from all files in corpus
    inverted_index = InvertedIndex()
    for filepath in filepaths:
        inverted_index.add_document(filepath)
    inverted_index.export_to_file(os.path.join(output_dir, 'inverted_index.txt'))


    # Create a tf-vector, prob-dist for each file
    # NOTE: you could skip this step, it's not needed to calculate the prob-dist for each category
    terms = inverted_index.get_terms()
    X_knn = []
    for filepath in filepaths:
        print(f'Creating tf-vector, prob-dist for file: {filepath}')
        tf_vector = inverted_index.get_tf_vector(filepath, pseudo_counts=1)
        X_knn.append(tf_vector)
        prob_dist = vector_to_prob_dist(tf_vector)

        output_filename = os.path.basename(filepath)
        output_filepath = os.path.join(output_dir, 'documents', output_filename)

        term_vec_to_file(terms, tf_vector, output_filepath.replace('.txt', '_tf.txt'))
        term_vec_to_file(terms, prob_dist, output_filepath.replace('.txt', '_prob_dist.txt'))

    # Create a tf-vector, prob-dist for each category
    for topic in topic_documents:
        print(f'Creating tf-vector, prob-dist for topic: {topic}')
        topic_tf_vector = np.zeros((len(terms)))

        files_in_topic = topic_documents[topic]
        for filepath in files_in_topic:
            tf_vector = inverted_index.get_tf_vector(filepath, pseudo_counts=1)
            topic_tf_vector = topic_tf_vector + tf_vector
        
        topic_prob_dist = vector_to_prob_dist(topic_tf_vector)

        output_filepath = os.path.join(output_dir, 'categories', f'{topic}.txt')
        term_vec_to_file(terms, topic_tf_vector, output_filepath.replace('.txt', '_tf.txt'))
        term_vec_to_file(terms, topic_prob_dist, output_filepath.replace('.txt', '_prob_dist.txt'))

    # KNN Classification
    print('Performing KNN classification with k=2...')
    X = X_knn
    # print(topic_documents)
    y = []
    for key, value in topic_documents.items():
        for i in value:
            y.append(list(topic_documents).index(key))
    # print(y)
    print('Classification Results:')
    X_test, knn = knn_classification(2, X, y)
    # print("x test", X_test, type(X_test))
    files = []
    for filepath in filepaths:
        tf_vector = inverted_index.get_tf_vector(filepath, pseudo_counts=1)
        # print("current tf vector", type(tf_vector))
        if np.any(np.all(tf_vector == X_test, axis=1)):
            files.append(filepath)
    # print("Files", files)
    print("Prediction for files", files, ":")
    pred = []
    for i in knn:
        pred.append(list(topic_documents)[i])
    print(pred)
   

@click.command()
@click.option('--issue-url', help='Full url to the issue on Github')
@click.option('--api-token', help='Github API token')
@click.option('--lm-dir', help='Folder with topic LMs, output of the "learn" step')
def classify(issue_url, api_token, lm_dir):
    """
    TODO: Download the issue at the given url, classify it using the given topic LMs, then output the result
    """
    print('TODO: unimplemented')


if __name__ == '__main__':
    cli.add_command(download)
    cli.add_command(learn)
    cli.add_command(classify)
    cli()
import click
import os
import numpy as np
from inverted_index import InvertedIndex
from prob_dist import vector_to_prob_dist
from util import term_vec_to_file
from tokenization import create_tf_dict
from similarity import cosine_similarity

topic_documents = {
    'animals': [
        './testdata/wikipedia/bird.txt',
        './testdata/wikipedia/cat.txt',
        './testdata/wikipedia/dog.txt',
        './testdata/wikipedia/fish.txt',
    ],
    'places': [
        './testdata/wikipedia/champaign.txt',
        './testdata/wikipedia/chicago.txt',
        './testdata/wikipedia/uiuc.txt',
    ],
}
# topic_documents = {
#     'corpus': [
#         './testdata/dummy/1.txt',
#         './testdata/dummy/2.txt',
#         './testdata/dummy/3.txt',
#     ],
#     'happy': [
#         './testdata/dummy/1.txt',
#         './testdata/dummy/3.txt',
#     ],
#     'sad': [
#         './testdata/dummy/2.txt',
#     ],
# }

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
@click.option('--output-dir', default='./output', help='Folder to put LMs into')
def learn(output_dir):
    """
    Do topic modeling on a set of input text files with known classifications.

    Estimates a ML Unigram LM for each topic given, and writes the LMs to an output directory.
    """
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

    # Create a tf-vector, prob-dist for each category
    for topic in topic_documents:
        print(f'Creating tf-vector, prob-dist for topic: {topic}')
        topic_tf_vector = np.zeros((len(terms)))

        files_in_topic = topic_documents[topic]
        for filepath in files_in_topic:
            doc_tf_vector = inverted_index.get_tf_vector(filepath, pseudo_counts=0)
            topic_tf_vector = topic_tf_vector + doc_tf_vector
            # apply tf-idf transformation
            transformed_vector = inverted_index.apply_tf_idf_transforms(topic_tf_vector)

        topic_lm = vector_to_prob_dist(topic_tf_vector)

        dir = f'./output/categories/{topic}'
        if not os.path.exists(dir):
            os.makedirs(dir)

        topic_prob_dist = vector_to_prob_dist(transformed_vector)
        output_filepath = os.path.join(output_dir, 'categories', f'{topic}.txt')
        term_vec_to_file(terms, transformed_vector, output_filepath.replace('.txt', '_tf.txt'))
        term_vec_to_file(terms, topic_prob_dist, output_filepath.replace('.txt', '_prob_dist.txt'))
        term_vec_to_file(terms, topic_tf_vector, os.path.join(dir, 'tf.txt'))
        np.save(os.path.join(dir, 'lm.npy'), topic_lm)
        np.save(os.path.join(dir, 'tf.npy'), topic_tf_vector)


@click.command()
@click.option('--filepath', help='File to classify')
def classify(filepath):
    if filepath is None:
        print("Please give a filepath with --filepath")
        return

    # Read the inverted index to get the list of terms (vector dimensions)
    inverted_index = InvertedIndex()
    inverted_index.load_from_file('./output/inverted_index.txt')

    # Parse the input file into a tf-dict, then convert it to a vector with the same dimensions as the inverted index
    doc_tf_dict = create_tf_dict(filepath)
    doc_tf_vector = inverted_index.tf_dict_to_vector(doc_tf_dict, pseudo_counts=0)
    doc_lm = vector_to_prob_dist(doc_tf_vector)

    # Compare the tf_vector to all our other vectors
    for topic in topic_documents:
        topic_prob_dist = np.load(f'./output/categories/{topic}/lm.npy')
        sim = cosine_similarity(doc_lm, topic_prob_dist)

        print(f'Similarity with topic {topic}: {sim}')


if __name__ == '__main__':
    cli.add_command(download)
    cli.add_command(learn)
    cli.add_command(classify)
    cli()
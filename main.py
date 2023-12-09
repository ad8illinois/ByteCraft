import click
import os
import numpy as np
from inverted_index import InvertedIndex
from prob_dist import vector_to_prob_dist
from util import term_vec_to_file
from tokenization import create_tf_dict
from similarity import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from github_issues_API import GithubClient

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
# }
topic_documents = {
    'happy': [
        './testdata/dummy/1.txt',
        './testdata/dummy/3.txt',
    ],
    'sad': [
        './testdata/dummy/2.txt',
    ],
}

@click.group()
def cli():
    pass

@click.command()
@click.option('--project-url', help='Full url to the project on github')
@click.option('--api-token', help='Github API token')
@click.option('--limit', type=click.INT,  help='Max number of issues to download')
def download(project_url, api_token, limit):
    if project_url is None or api_token is None or limit is None:
        print('Please provide all CLI options --project-url --api-token --limit')
        return

    if project_url.endswith('/'): # Remove any trailing slashes
        project_url[:len(project_url)-2]
    
    owner = project_url.split('/')[-2]
    repo = project_url.split('/')[-1]
    print(f'Owner: {owner}, Repo: {repo}')

    github = GithubClient(token=api_token)
    top_contibutors = github.get_top_contributors(owner, repo)
    print(f"Top contributors: {top_contibutors}")

    relevant_issues = github.get_issues_commented_by_top_contributors(owner, repo, top_contibutors, limit)

    # print(f"Current open issues: {len(open_issues)}")
    print(f"Issues that the top 5 contributors have commented on: {[i['number'] for i in relevant_issues]}")

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

    all_filepaths = []
    filepaths = []
    for topic in topic_documents:
        for filepath in topic_documents[topic]:
            all_filepaths.append(filepath)
            filepaths.append(filepath)
    
    # Construct inverted index from all files in corpus
    inverted_index = InvertedIndex()
    for filepath in filepaths:
        inverted_index.add_document(filepath)
    inverted_index.export_to_file(os.path.join(output_dir, 'inverted_index.txt'))
    terms = inverted_index.get_terms()

    vectorizer = TfidfVectorizer(input='filename', vocabulary=terms)
    tf_idf_matrix = vectorizer.fit_transform(all_filepaths)
    print(tf_idf_matrix)
        # TODO: Somehow get each row out of tf_idf_matrix, into a numpy array that we can run our other code on
        # TODO: Also, verify that it's actually working, common words are getting lower weights compared to the raw tf_vectors

    # Create a tf-vector, prob-dist for each category
    for topic in topic_documents:
        print(f'Creating tf-vector, prob-dist for topic: {topic}')
        topic_tf_vector = np.zeros((len(terms)))

        files_in_topic = topic_documents[topic]
        for filepath in files_in_topic:
            doc_tf_vector = inverted_index.get_tf_vector(filepath, pseudo_counts=0)
            topic_tf_vector = topic_tf_vector + doc_tf_vector
            # apply tf-idf transformation
            topic_tf_idf_vector = inverted_index.apply_tf_idf_transformation(topic_tf_vector)

        topic_tf_idf_lm = vector_to_prob_dist(topic_tf_idf_vector)
        topic_tf_lm = vector_to_prob_dist(topic_tf_vector)

        dir = f'./output/categories/{topic}'
        if not os.path.exists(dir):
            os.makedirs(dir)


        term_vec_to_file(terms, topic_tf_vector, os.path.join(dir, 'tf.txt'))
        term_vec_to_file(terms, topic_tf_lm, os.path.join(dir, 'tf_lm.txt'))
        term_vec_to_file(terms, topic_tf_idf_vector, os.path.join(dir, 'tf_idf.txt'))
        term_vec_to_file(terms, topic_tf_idf_lm, os.path.join(dir, 'tf_idf_lm.txt'))

        np.save(os.path.join(dir, 'tf.npy'), topic_tf_vector)
        np.save(os.path.join(dir, 'tf_lm.npy'), topic_tf_lm)
        np.save(os.path.join(dir, 'tf_idf.npy'), topic_tf_vector)
        np.save(os.path.join(dir, 'tf_idf_lm.npy'), topic_tf_lm)


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
        topic_prob_dist = np.load(f'./output/categories/{topic}/tf_idf.npy')
        sim = cosine_similarity(doc_lm, topic_prob_dist)

        print(f'Similarity with topic {topic}: {sim}')


if __name__ == '__main__':
    cli.add_command(download)
    cli.add_command(learn)
    cli.add_command(classify)
    cli()
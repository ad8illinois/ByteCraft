import random
from collections import Counter
import click
import tempfile
from pathlib import Path
import os
import numpy as np
from inverted_index import InvertedIndex
import json
from util import term_vec_to_file
from tokenization import create_tf_dict, count_total_tokens_in_doc
from issue_fetcher import IssueFetcher
from similarity import euclidian_distance, cosine_similarity
from github_issues_API import GithubClient
from ml_model_definitions import knn_classification, naive_bayes_classification
import ast
from urllib.parse import urlparse


@click.group()
def cli():
    pass

@click.command()
@click.option('--data-dir', help='Folder to store issues, vectors')
@click.option('--api-token', help='Github API token')
@click.option('--project-url', help='Full url to the project on github')
@click.option('--users', help='Comma-separated list of users. Defaults to top 5 contributors')
@click.option('--limit', type=click.INT, default=500, help='Max number of issues to download. Defaults to 500')
def download(data_dir, api_token, project_url, users, limit):
    if data_dir is None or api_token is None or project_url is None:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    issues_dir = os.path.join(data_dir, 'issues/')
    if not os.path.exists(issues_dir):
        os.makedirs(os.path.join(issues_dir))

    if project_url.endswith('/'): # Remove any trailing slashes
        project_url[:len(project_url)-2]
    
    owner = project_url.split('/')[-2]
    repo = project_url.split('/')[-1]
    print(f'Owner: {owner}, Repo: {repo}')

    github = GithubClient(token=api_token, owner=owner, repo=repo)

    if users is not None:
        relevant_users = users.split(',')
    else:
        relevant_users = github.get_top_contributors()
        print(f"No users provided. Using top 5 contributors: {relevant_users}")

    fetcher = IssueFetcher(output_dir=issues_dir, owner=owner, repo=repo, token=api_token)
    issue_index = fetcher.fetch_issues_for_users(limit, relevant_users)

    index_filepath = os.path.join(data_dir, 'index.json')
    print(f'Writing issue index to {index_filepath}')
    with open(index_filepath, 'w') as file:
        file.write(json.dumps(issue_index, indent=2))

@click.command()
@click.option('--data-dir', help='Folder to store issues, vectors')
def learn(data_dir):
    """
    Index 80% of the files in the corpus, and create tf-idf vector representations for all files.

    Run classification on 20% of the files, calculate accuracy.
    """
    if data_dir is None:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    documents_dir = os.path.join(data_dir, 'documents/')
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
    categories_dir = os.path.join(data_dir, 'categories/')
    if not os.path.exists(categories_dir):
        os.makedirs(categories_dir)

    index_filepath = os.path.join(data_dir, 'index.json')
    with open(index_filepath, 'r') as file:
        topic_documents = json.loads(file.read())

    # Create training and evaluation datasets. 80% in training, 20% in evaluation.
    training_topic_documents = {}
    evaluation_topic_documents = {}
    for topic in topic_documents:
        docs = topic_documents[topic]
        training = random.sample(docs, k=int(len(docs) * .8))
        evaluation = [d for d in docs if d not in training]

        training_topic_documents[topic] = training
        evaluation_topic_documents[topic] = evaluation

    with open(os.path.join(data_dir, 'training.json'), 'w') as file:
        file.write(json.dumps(training_topic_documents, indent=2))
    with open(os.path.join(data_dir, 'evaluation.json'), 'w') as file:
        file.write(json.dumps(evaluation_topic_documents, indent=2))

    training_filepaths = []
    for topic in training_topic_documents:
        for filepath in training_topic_documents[topic]:
            training_filepaths.append(filepath)
    
    # Construct inverted index from all files in corpus
    inverted_index = InvertedIndex()
    for filepath in training_filepaths:
        inverted_index.add_document(filepath)
        inverted_index.total_terms_in_doc[filepath] = count_total_tokens_in_doc(filepath)
    inverted_index.export_to_file(os.path.join(data_dir, 'inverted_index.txt'))
    terms = inverted_index.get_terms()

    # Write TF-IDF vectors for each file to the output_dir
    for filepath in training_filepaths:
        print(f'Generating tf-idf for doc {filepath}')
        stem = Path(filepath).stem

        doc_tf = inverted_index.get_tf_vector(filepath)
        doc_tf_idf = inverted_index.apply_idf(inverted_index.apply_tf_transformation(doc_tf))

        term_vec_to_file(terms, doc_tf, os.path.join(data_dir, 'documents', stem + '_tf.txt'))
        np.save(os.path.join(data_dir, 'documents', stem + '_tf.npy'), doc_tf)

        # doc_tfidf_vector = inverted_index.compute_tf_idf_vector()
        # np.save(os.path.join(data_dir, 'documents', stem + '_tf_idf.npy'), doc_tfidf_vector)
        # np.save(os.path.join(data_dir, 'tf-idf-transformation-vector.npy'), doc_tfidf_vector)

        # Use this for the naive implementation of tf-idf
        term_vec_to_file(terms, doc_tf_idf, os.path.join(documents_dir, stem + '_tf_idf.txt'))
        np.save(os.path.join(data_dir, 'documents', stem + '_tf_idf.npy'), doc_tf_idf)

    print('-----------')
    print('Evaluation')
    print('-----------')
    correct = 0
    total = 0
    topic_results = {}
    for topic in evaluation_topic_documents:
        topic_results[topic]={
            'correct': 0,
            'total': 0,
        }
        for filepath in evaluation_topic_documents[topic]:
            predicted_topic = classify_file(
                data_dir=data_dir,
                filepath=filepath,
                method='knn',
                verbose=False
            )
            print(f'{filepath} - User {topic} - Predicted User {predicted_topic}')
            if topic == predicted_topic:
                correct = correct + 1
                topic_results[topic]['correct']+= 1
            total = total + 1
            topic_results[topic]['total']+= 1
    print('-----------')
    print('Evaluation Accuracy')
    for t in topic_results:
        t_correct=topic_results[t]['correct']
        t_total=topic_results[t]['total']
        print(f"  {t}: {t_correct} / {t_total} = {t_correct / t_total}")
    print(f'  overall: {correct} / {total} = {correct / total}')
    print('-----------')

@click.command()
@click.option('--data-dir', help='Folder to store issues, vectors')
@click.option('--method', help='Either "knn" or "naivebayes"')
@click.option('--api-token', help='Github API token')
@click.option('--github-issue', help='GitHub issue URL or issue number to classify')
@click.option('--filepath', help='File to classify. If given, ignores github-issue and api-token')
@click.option('--verbose', is_flag=True, type=click.BOOL, default=False, help='If defined, ')
def classify(data_dir, method, api_token, github_issue, filepath, verbose):
    if data_dir is None or method is None:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()
    
    if github_issue is not None and api_token is not None:
        # Extract owner, repo, and issue number from the GitHub issue URL
        parsed_url = urlparse(github_issue)
        path_parts = parsed_url.path.split('/')
        if len(path_parts) < 3:
            print('Invalid GitHub issue URL. Please provide a valid GitHub issue URL.')
            return
        owner = path_parts[1]
        repo = path_parts[2]
        issue_number = path_parts[-1]

        # Download the issue to a temporary file
        fetcher = IssueFetcher(owner, repo, api_token, output_dir=tempfile.gettempdir())
        filepath = fetcher.fetch_issue(issue_number)

    if filepath is None:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    predicted_topic = classify_file(data_dir, filepath, method, verbose)
    print('Classification Results:', predicted_topic)

    # Issue duplication detection
    duplicates = find_duplicates(data_dir, filepath,)
    print('Duplicate docs: ', duplicates)
   

def find_duplicates(data_dir, filepath):
    inverted_index = InvertedIndex()
    inverted_index.load_from_file(os.path.join(data_dir, 'inverted_index.txt'))
    doc_tf_dict = create_tf_dict(filepath)
    doc_tf_vector = inverted_index.tf_dict_to_vector(doc_tf_dict, pseudo_counts=0) # TODO: replace with tf-idf
    doc_tfidf_vector = inverted_index.apply_idf(inverted_index.apply_tf_transformation(doc_tf_vector))
    # doc_tfidf_vector= np.load(os.path.join(data_dir, 'tf-idf-transformation-vector.npy'))

    similarity_threshold = 0.9  # Adjust the threshold as needed
    duplicate_found = False
    duplicate_issue = None

    # Load the vectors from all previously-learned files
    with open(os.path.join(data_dir, 'training.json'), 'r') as file:
        topic_documents = json.loads(file.read()) 

    # Shape input data into KNN-compatible format
    training_doc_filenames = []
    training_docs = []
    topic_index_map = {}  # Maps from the topic index back to the topic itself
    for topic_index, topic in enumerate(topic_documents):
        topic_index_map[topic_index] = topic
        files_in_topic = topic_documents[topic]
        for training_doc_filename in files_in_topic:
            stem = Path(training_doc_filename).stem
            training_doc_tfidf_vector = np.load(os.path.join(data_dir, 'documents', f'{stem}_tf_idf.npy'))

            training_doc_filenames.append(training_doc_filename)
            training_docs.append(training_doc_tfidf_vector)

    duplicate_docs = []
    for i, training_doc in enumerate(training_docs):
        similarity_score = cosine_similarity(training_doc, doc_tfidf_vector)
        if similarity_score > similarity_threshold:
            duplicate_issue = training_doc_filenames[i]
            duplicate_docs.append(duplicate_issue)
            print(f"Duplicate issue found: {duplicate_issue} {similarity_score}")

    return duplicate_docs

def classify_file(data_dir, filepath, method, verbose=False):
    # Read the inverted index, use it to get a tf-vector for the new input file
    inverted_index = InvertedIndex()
    inverted_index.load_from_file(os.path.join(data_dir, 'inverted_index.txt'))

    doc_tf_dict = create_tf_dict(filepath)
    doc_tf_vector = inverted_index.tf_dict_to_vector(doc_tf_dict, pseudo_counts=0) # TODO: replace with tf-idf
    doc_tfidf_vector = inverted_index.apply_idf(inverted_index.apply_tf_transformation(doc_tf_vector))

    # Load the vectors from all previously-learned files
    with open(os.path.join(data_dir, 'training.json'), 'r') as file:
        topic_documents = json.loads(file.read()) 

    training_doc_filenames = []
    training_docs = []
    training_labels = []
    topic_index_map = {}  # Maps from the topic index back to the topic itself
    for topic_index, topic in enumerate(topic_documents):
        topic_index_map[topic_index] = topic
        files_in_topic = topic_documents[topic]
        for training_doc_filename in files_in_topic:

            stem = Path(training_doc_filename).stem
            training_doc_tfidf_vector = np.load(os.path.join(data_dir, 'documents', f'{stem}_tf_idf.npy'))

            training_doc_filenames.append(training_doc_filename)
            training_docs.append(training_doc_tfidf_vector)
            training_labels.append(topic_index)

    if method == 'knn':
        distances = []
        for i, training_doc in enumerate(training_docs):
            distances.append({
                'filename': training_doc_filenames[i],
                'topic': topic_index_map[training_labels[i]],
                'similarity': cosine_similarity(training_doc, doc_tfidf_vector),
            })
        # distances = sorted(distances, key=lambda d: d['similarity'][0], reverse=True)
        # Comment the line above and use this for the other naive implementation of tf-idf
        distances = sorted(distances, key=lambda d: d['similarity'], reverse=True)
        top_n = distances[:5]
        top_n_topics = [d['topic'] for d in top_n]
        most_common = Counter(top_n_topics).most_common()
        predicted_topic = most_common[0][0]

        if verbose:
            for d in top_n:
                print(f"Filepath: {d['filename']}  Topic: {d['topic']}  Similarity: {d['similarity']}")

        return predicted_topic
    
    if method == 'naivebayes':
        nb = naive_bayes_classification(training_docs, training_labels, [doc_tfidf_vector])
        predicted_topic_index = nb[0]
        predicted_topic = topic_index_map[predicted_topic_index]
        return predicted_topic



if __name__ == '__main__':
    cli.add_command(download)
    cli.add_command(learn)
    cli.add_command(classify)
    cli()





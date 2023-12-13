import click
from pathlib import Path
import os
import numpy as np
from inverted_index import InvertedIndex
from prob_dist import vector_to_prob_dist
import json
from util import term_vec_to_file
from tokenization import create_tf_dict
from issue_fetcher import IssueFetcher
from similarity import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from github_issues_API import GithubClient
from scipy.sparse import csr_matrix
from ml_model_definitions import knn_classification
import ast
from urllib.parse import urlparse


@click.group()
def cli():
    pass

@click.command()
@click.option('--output-dir', help='Folder to store issues in')
@click.option('--api-token', help='Github API token')
@click.option('--project-url', help='Full url to the project on github')
@click.option('--users', help='Comma-separated list of users. Defaults to top 5 contributors')
@click.option('--limit', type=click.INT, default=500, help='Max number of issues to download. Defaults to 500')
def download(output_dir, api_token, project_url, users, limit):
    if output_dir is None or api_token is None or project_url is None:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    fetcher = IssueFetcher(output_dir=output_dir, owner=owner, repo=repo, token=api_token)
    issue_index = fetcher.fetch_issues_for_users(limit, relevant_users)

    index_filepath = os.path.join(output_dir, 'index.json')
    print(f'Writing issue index to {index_filepath}')
    with open(index_filepath, 'w') as file:
        file.write(json.dumps(issue_index, indent=2))

@click.command()
@click.option('--index-file', help='Path to index file')
@click.option('--output-dir', help='Folder to store output files')
def learn(index_file, output_dir):
    """
    Do topic modeling on a set of input text files with known classifications.

    Estimates a ML Unigram LM for each topic given, and writes the LMs to an output directory.
    """
    if index_file is None or output_dir is None:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'documents')):
        os.makedirs(os.path.join(output_dir, 'documents'))
    if not os.path.exists(os.path.join(output_dir, 'categories')):
        os.makedirs(os.path.join(output_dir, 'categories'))
    
    with open(index_file, 'r') as file:
        topic_documents = json.loads(file.read())

    with open(os.path.join(output_dir, 'index.json'), 'w') as file:
        file.write(json.dumps(topic_documents, indent=2))

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
    raw_matrix = vectorizer.fit_transform(all_filepaths)

    # Get each row out of tf_idf_matrix, into a numpy array that we can run our other code on
    # TODO: Also, verify that it's actually working, common words are getting lower weights compared to the raw tf_vectors

    sparse_matrix = csr_matrix(raw_matrix, dtype=float)
    dense_tf_idf_matrix = sparse_matrix.toarray()

    tf_idf_matrix_np_rows = [np.array(row) for row in dense_tf_idf_matrix]

    # Write TF-IDF vectors for each file to the output_dir
    for i, row in enumerate(tf_idf_matrix_np_rows):
        original_filepath = all_filepaths[i]
        stem = Path(original_filepath).stem

        doc_tf = inverted_index.get_tf_vector(original_filepath)
        doc_tf_idf = vector_to_prob_dist(row)

        term_vec_to_file(terms, doc_tf, os.path.join(output_dir, 'documents', stem + '_tf.txt'))
        np.save(os.path.join(output_dir, 'documents', stem + '_tf.npy'), doc_tf)
        
        term_vec_to_file(terms, doc_tf_idf, os.path.join(output_dir, 'documents', stem + '_tf_idf.txt'))
        np.save(os.path.join(output_dir, 'documents', stem + '_tf_idf.npy'), doc_tf_idf)

    # Create a tf-vector, prob-dist for each category
    for topic in topic_documents:
        print(f'Creating tf-vector, prob-dist for topic: {topic}')
        topic_tf_vector = np.zeros((len(terms)))

        files_in_topic = topic_documents[topic]
        for filepath in files_in_topic:
            doc_tf_vector = inverted_index.get_tf_vector(filepath, pseudo_counts=0)
            topic_tf_vector = topic_tf_vector + doc_tf_vector

            """
            TODO: This topic_tf_idf_vector is not accurate.
            It doesn't count doc-frequency for the entire corpus, just within this topic
            """
            topic_tf_idf_vector = inverted_index.compute_tf_idf_transformation(topic_tf_vector)

        topic_tf_idf_lm = vector_to_prob_dist(topic_tf_idf_vector)
        topic_tf_lm = vector_to_prob_dist(topic_tf_vector)

        dir = os.path.join(output_dir, 'categories', topic)
        if not os.path.exists(dir):
            os.makedirs(dir)

        term_vec_to_file(terms, topic_tf_vector, os.path.join(dir, 'tf.txt'))
        term_vec_to_file(terms, topic_tf_lm, os.path.join(dir, 'tf_lm.txt'))
        term_vec_to_file(terms, topic_tf_idf_vector, os.path.join(dir, 'tf_idf.txt'))
        term_vec_to_file(terms, topic_tf_idf_lm, os.path.join(dir, 'tf_idf_lm.txt'))

        np.save(os.path.join(dir, 'tf.npy'), topic_tf_vector)
        np.save(os.path.join(dir, 'tf_lm.npy'), topic_tf_lm)
        np.save(os.path.join(dir, 'tf_idf.npy'), topic_tf_idf_vector)
        np.save(os.path.join(dir, 'tf_idf_lm.npy'), topic_tf_idf_lm)


# @click.command()
# @click.option('--learn-dir', help='Folder generated from the "learn" step')
# @click.option('--filepath', help='File to classify')
# def classify(learn_dir, filepath):
#     if filepath is None or learn_dir is None:
#         ctx = click.get_current_context()
#         click.echo(ctx.get_help())
#         ctx.exit()

#     # Load the index, which has the categories as keys
#     with open(os.path.join(learn_dir, 'index.json'), 'r') as file:
#         topic_documents = json.loads(file.read()) 

#     # Read the inverted index to get the list of terms (vector dimensions)
#     inverted_index = InvertedIndex()
#     inverted_index.load_from_file(os.path.join(learn_dir, 'inverted_index.txt'))

#     # Parse the input file into a tf-dict, then convert it to a vector with the same dimensions as the inverted index
#     doc_tf_dict = create_tf_dict(filepath)
#     doc_tf_vector = inverted_index.tf_dict_to_vector(doc_tf_dict, pseudo_counts=0)
#     doc_lm = vector_to_prob_dist(doc_tf_vector)

#     """
#     TODO:  
#         Right now, the classification just compares the input file 
#         with the TF-vector of each category.

#         We need to replace this with actual KNN or Naive Bayes
#     """

#     # KNN Classification
#     X_knn = list()
#     for topic in topic_documents:
#         topic_tf_vector = np.zeros((len(inverted_index.get_terms())))
#         files_in_topic = topic_documents[topic]
#         for f in files_in_topic:
#             # NOTE for Annamika: you can load the tf-idf vector for any file from <learn_dir>/documents/<file_root>_tf_idf.npy
#             doc_tf_vector = inverted_index.get_tf_vector(f, pseudo_counts=0)
#             topic_tf_vector = topic_tf_vector + doc_tf_vector
#             X_knn.append(doc_tf_vector)

#     print('Performing KNN classification with k=2...')
#     X = X_knn
#     y = []
#     for key, value in topic_documents.items():
#         for i in value:
#             y.append(list(topic_documents).index(key))
    
#     doc_tf_dict = create_tf_dict(filepath)
#     doc_tf_vector = inverted_index.tf_dict_to_vector(doc_tf_dict, pseudo_counts=0)

#     knn = knn_classification(2, X, y, [doc_tf_vector])
#     print('KNN Classification Results:')
#     pred = []
#     for i in knn:
#         pred.append(list(topic_documents)[i])
#     print("Prediction for file:", filepath, ' --> ', pred[0])
#     print('')

#     # Compare the tf_vector to all our other vectors
#     sim_results = []
#     for topic in topic_documents:
#         # TODO: replace this with the tf-idf vector, instead of just tf
#         topic_prob_dist = np.load(os.path.join(learn_dir, 'categories', topic, 'tf_lm.npy'))
#         sim = cosine_similarity(doc_lm, topic_prob_dist)
#         sim_results.append({
#             'category': topic,
#             'similarity': sim,
#         })
    
#     sim_results = sorted(sim_results, key=lambda sr: sr['similarity'], reverse=True)
#     for rs in sim_results:
#         print(f'Similarity with category {rs["category"]}: {rs["similarity"]}')


# if __name__ == '__main__':
#     cli.add_command(download)
#     cli.add_command(learn)
#     cli.add_command(classify)
#     cli()




# Classify using URL
# Same as the function above except it takes github URL for an issue and downloads it instead (can get rid of the above one if this works the same)

# This function is used to download the new issue 
def download_new_issue(output_dir, owner, repo, issue_number, token):
   # Fetch the issue details from GitHub
    github = GithubClient(token, owner, repo)
    issue_details = github.get_issue_details(issue_number)
    comments = github.get_comments(issue_number)
    reporter = issue_details['user']['login']

    # Write the fetched issue details to a new issue file in same directory and add it to index
    filepath = os.path.join(output_dir, f'{issue_number}.txt')
    print(f'Saving issue {issue_number}. Filepath {filepath}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(filepath, 'a', encoding='utf-8') as file:
        file.write('----------------\n')
        file.write(f"Issue: #{issue_number}\n")
        file.write(f"Issue title: {issue_details['title']}\n")
        file.write(f"Issue url: {issue_details['html_url']}\n")
        file.write(f"Reporter: {reporter}\n")
        file.write('----------------\n')
        file.write(f'{issue_details["body"]}\n')
        file.write('\n\n')
        for comment in comments:
            file.write('----------------\n')
            file.write(f'Comment\n')
            file.write(f"User: {comment['user']['login']}\n")
            file.write('----------------\n')
            file.write(f'{comment["body"]}\n')
            file.write('\n\n')
    return filepath

@click.command()
@click.option('--learn-dir', help='Folder generated from the "learn" step')
@click.option('--output-dir', help='Folder where new issue will be stored, should be same as the output_dir used in the "download" step')
@click.option('--api-token', help='Github API token')
@click.option('--github-issue', help='GitHub issue URL or issue number to classify')
def classify(learn_dir, output_dir, api_token, github_issue):
    if github_issue is None or learn_dir is None:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    # Extract owner, repo, and issue number from the GitHub issue URL
    parsed_url = urlparse(github_issue)
    
    path_parts = parsed_url.path.split('/')
    if len(path_parts) < 3:
        print('Invalid GitHub issue URL. Please provide a valid GitHub issue URL.')
        return
    
    owner = path_parts[1]
    repo = path_parts[2]
    issue_number = path_parts[-1]

    # Load the index, which has the categories as keys
    with open(os.path.join(learn_dir, 'index.json'), 'r') as file:
        topic_documents = json.loads(file.read()) 

    filepath = download_new_issue(output_dir, owner, repo, issue_number, api_token)

    # Read the inverted index to get the list of terms (vector dimensions)
    inverted_index = InvertedIndex()
    inverted_index.load_from_file(os.path.join(learn_dir, 'inverted_index.txt'))
    
    # Parse the input file into a tf-dict, then convert it to a vector with the same dimensions as the inverted index
    doc_tf_dict = create_tf_dict(filepath)
    doc_tf_vector = inverted_index.tf_dict_to_vector(doc_tf_dict, pseudo_counts=0)
    doc_lm = vector_to_prob_dist(doc_tf_vector)


    """
    TODO:  
        Right now, the classification just compares the input file 
        with the TF-vector of each category.

        We need to replace this with actual KNN or Naive Bayes
    """

    # KNN Classification
    X_knn = list()
    for topic in topic_documents:
        topic_tf_vector = np.zeros((len(inverted_index.get_terms())))
        files_in_topic = topic_documents[topic]
        for f in files_in_topic:
            # NOTE for Annamika: you can load the tf-idf vector for any file from <learn_dir>/documents/<file_root>_tf_idf.npy
            doc_tf_vector = inverted_index.get_tf_vector(f, pseudo_counts=0)
            topic_tf_vector = topic_tf_vector + doc_tf_vector
            X_knn.append(doc_tf_vector)

    print('Performing KNN classification with k=2...')
    X = X_knn
    y = []
    for key, value in topic_documents.items():
        for i in value:
            y.append(list(topic_documents).index(key))
    
    doc_tf_dict = create_tf_dict(filepath)
    doc_tf_vector = inverted_index.tf_dict_to_vector(doc_tf_dict, pseudo_counts=0)

    knn = knn_classification(2, X, y, [doc_tf_vector])
    print('KNN Classification Results:')
    pred = []
    for i in knn:
        pred.append(list(topic_documents)[i])
    print("Prediction for file:", filepath, ' --> ', pred[0])
    print('')

    # Compare the tf_vector to all our other vectors
    sim_results = []
    for topic in topic_documents:
        # TODO: replace this with the tf-idf vector, instead of just tf
        topic_prob_dist = np.load(os.path.join(learn_dir, 'categories', topic, 'tf_lm.npy'))
        sim = cosine_similarity(doc_lm, topic_prob_dist)
        sim_results.append({
            'category': topic,
            'similarity': sim,
        })
    
    sim_results = sorted(sim_results, key=lambda sr: sr['similarity'], reverse=True)
    for rs in sim_results:
        print(f'Similarity with category {rs["category"]}: {rs["similarity"]}')


if __name__ == '__main__':
    cli.add_command(download)
    cli.add_command(learn)
    cli.add_command(classify)
    cli()





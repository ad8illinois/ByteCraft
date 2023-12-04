import click
import os
import tokenization as tk

@click.group()
def cli():
    pass

@click.command()
@click.option('--project-url', help='Full url to the project on github')
@click.option('--api-token', help='Github API token ')
@click.option('--dir', help='Download location')
def download():
    """
    TODO: Download all the issues from a github repo, and save them into a local folder
    """
    print('TODO: unimplemented')

@click.command()
@click.option('--input-dir', help='Folder with training data')
@click.option('--output-dir', default='./output', help='Folder to put LMs into')
def learn(output_dir):
    """
    Do topic modeling on a set of input text files with known classifications.

    Estimates a ML Unigram LM for each topic given, and writes the LMs to an output directory.
    """
    # tokens, lms = tk.learn_topics({
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
    # })

    tokens, lms = tk.learn_topics({
        'corpus': [
            './testdata/dummy/1.txt',
            './testdata/dummy/2.txt',
            './testdata/dummy/3.txt',
        ],
        'happy': [
            './testdata/dummy/1.txt',
            './testdata/dummy/3.txt',
        ],
        'sad': [
            './testdata/dummy/2.txt',
        ],
    })

    # TODO: do some heurustics on input-data
    #    - Then let users just specify a list of folders. Each folder will become one topic
    #    - Automatically generate the 'corpus' topic, by combining all files into that topic

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for topic in lms:
        filepath = os.path.join(output_dir, f'{topic}.txt')
        tk.word_vec_to_file(tokens, lms[topic], filepath)
   

@click.command()
@click.option('--issue-url', help='Full url to the issue on Github')
@click.option('--api-token', help='Github API token')
@click.option('--lm-dir', help='Folder with topic LMs, output of the "learn" step')
def classify():
    """
    TODO: Download the issue at the given url, classify it using the given topic LMs, then output the result
    """
    print('Running classification...')


if __name__ == '__main__':
    cli.add_command(download)
    cli.add_command(learn)
    cli.add_command(classify)
    cli()
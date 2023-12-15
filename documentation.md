# ByteCraft Documentation

## Overview of the Function of the Code
VCS platforms like Github and Gitlab serve a variety of key functions in the open source community - code versioning, issue tracking, project documentation, communication, automation, and more. 
However, the size and scope of many open source projects today creates significant challenges in the area of task management and issue tracking.

This project is a command-line utility for triage of github issues.

The function of the code is to use Text Retrieval algorithms to triage new Github issues to the project maintainer who has the most knowledge on that issue, while also detecting duplicate or high-similarity issues in a large project. 

### Overview of Functions
- [github_issues_API.py](https://github.com/ad8illinois/ByteCraft/blob/master/src/github_issues_API.py): Functions to connect with the GitHub REST APIs 
- [inverted_index.py](https://github.com/ad8illinois/ByteCraft/blob/master/src/inverted_index.py): IDF and TF-IDF computations 
- [issue_fetcher.py](https://github.com/ad8illinois/ByteCraft/blob/master/src/issue_fetcher.py): Downloading and writing issues to files 
- [main.py](https://github.com/ad8illinois/ByteCraft/blob/master/src/main.py): Functions to run Download, Learn, and Classify. Assigns user to new issues and alerts of any duplicates
- [ml_model_definitions](https://github.com/ad8illinois/ByteCraft/blob/master/src/ml_model_definitions.py): KNN and Naive Bayes classifier functions
- [prob_dist.py](https://github.com/ad8illinois/ByteCraft/blob/master/src/prob_dist.py): Converting TF vectors to probability distributions
- [similarity.py](https://github.com/ad8illinois/ByteCraft/blob/master/src/similarity.py): Cosine Similarity and Euclidian Distance functions
- [tokenization.py](https://github.com/ad8illinois/ByteCraft/blob/master/src/tokenization.py): Tokenizing the files using NLTK
- [util.py](https://github.com/ad8illinois/ByteCraft/blob/master/src/util.py): Utility functions to write word vectors to files

## Implementation
We used Python for our project, because of the availability of libraries common in NLP and TR.

### Packages used: 
- numpy [NumPy](https://numpy.org/doc/)
- nltk [NLTK](https://www.nltk.org/index.html)
- sklearn [Scikit-learn](https://scikit-learn.org/stable/)
- requests [Requests](https://pypi.org/project/requests/)
- click [Click](https://click.palletsprojects.com/en/8.1.x/)
- pathlib [pathlib](https://docs.python.org/3/library/pathlib.html)
- urllib [urllib](https://docs.python.org/3/library/urllib.html)

### Classification Algorithms used:
- KNN [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- Naive Bayes [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)

### Features

#### Download: The first step in the classification system is downloading historical issues from a Github repo, which becomes our training dataset.
  The download step performs these tasks:
  1. Find the top 5 contributors for a repo (if no users are provided)
  2. Look at the N most recent issues in the repo
  3. Download any issues which are relevant to the given users
  4. Create an `index.json` file, mapping issue txt files to their respective user (category)

  Issues will be downloaded to the `./issues` folder, with the filename `./issues/<issue_number>.txt`.

  Additionally, an index file will be created at `./issues/index.json`, which maps from users to the issues relevant to that user

  Example output folder structure:
  ```
  $DATA_DIR/
    - issues/
        - 20670.txt  # Full text and comments of issue 20670
        - 20672.txt
        - 20676.txt
        - 20678.txt
    - index.json  # Index of all issue files, maps each file to a relevant user
  ```

#### Learn: The second step in classification is mapping documents into vector representations which we can compare.  
  In the learning step, we:
  1. Tokenize all input files, and initialize bag-of-words vector dimensions
  2. Create a term-frequency inverted-index for all files in the corpus
  3. Calculate TF-IDF vectors for each document
  4. Trial-run our classification algorithms, calculate accuracy for our training data-set

  This learning step ONLY indexes 80% of the files in the corpus (selected randomly). The remaining 20% are used as an evaluation dataset.

  The output files generated should look like:
  ```
  $DATA_DIR/
    - documents/
      - $ISSUE_NUMBER_tf.txt      # Count of all terms in this issue
      - $ISSUE_NUMBER_tf_idf.txt  # Count of terms, with IDF weighting applied
      ...
      - $FILENAME.npy             # Numpy-compatible serialization of the .txt file with the same name
    - inverted_index.txt          # Term-Frequency inverted index
    - training.json               # Index of files used for training
    - evaluation.json             # Index of files used for evaluation
  ```

#### Classify: The final step is to classify the issues into one of the learned categories
  In the Classify step, we:
  1. Download a new github issue
  2. Tokenize the issue, create a bag-of-words vector with the same dimensions as our training data
  3. Use one of our classification algorithms to classify it into one of the learned categories
      - KNN
      - Naive Bayes
    
  The output should look like:
  Filepath: ./data/issues/503.txt  Topic: NetanelBasal  Similarity: 1.0
  Filepath: ./data/issues/496.txt  Topic: NetanelBasal  Similarity: 0.0915722322675805
  Filepath: ./data/issues/493.txt  Topic: EricPoul  Similarity: 0.08406253459296453
  Filepath: ./data/issues/490.txt  Topic: NetanelBasal  Similarity: 0.014407300373578137
  Classification Results: NetanelBasal
  Duplicate issue found: ./data/issues/503.txt 1.0
  Duplicate docs:  ['./data/issues/503.txt']

## Usage

### Setup and Dependencies
Native python (recommended)
```
python -m venv venv
source venv/bin/activate

pip install -r ./requirements.txt
python -c "import nltk; nltk.download('popular')"
```

Docker
```
docker build -t github-triage .

# In any further docs in this file, instead of:
  python src/main.py <command> --data-dir ./data ...

# Write:
  docker run -v $PWD/data:/data github-triage <command> --data-dir /data ...
```

### Quick Start - Example
Before you start, you will need to generate a Github Personal Access Token. Docs can be found here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
```
GITHUB_API_TOKEN='replaceme'
```

Step 1: Download issues from Github
```
$ python ./src/main.py download \
    --data-dir ./data \
    --api-token $GITHUB_API_TOKEN \
    --users NetanelBasal,shaharkazaz,ido-golan,EricPoul \
    --project-url 'https://github.com/ngneat/elf'
```

Step 2: Preprocess training dataset
```
python src/main.py learn --data-dir ./data
```

Step 3: Classify new issues (also looks for duplicate issues)
```
python ./src/main.py classify \
    --data-dir ./data \
    --api-token $GITHUB_API_TOKEN \
    --github-issue  https://github.com/ngneat/elf/issues/503
```
For a more detailed explanation of what each command does, keep reading below.

### Step 1 - Download training data
The first step in the classification system is downloading historical issues from a Github repo, which becomes our training dataset. 

To download issues, generate a Github Personal Access Token following the instructions here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens

The download step performs these tasks:
1. Find the top 5 contributors for a repo (if no users are provided)
2. Look at the N most recent issues in the repo
3. Download any issues which are relevant to the given users
4. Create an `index.json` file, mapping issue txt files to their respective user (category)

To run the download step, use the `download` command:
```
% python ./src/main.py download
Usage: main.py download [OPTIONS]

Options:
  --data-dir TEXT     Folder to store issues, vectors
  --api-token TEXT    Github API token
  --project-url TEXT  Full url to the project on github
  --users TEXT        Comma-separated list of users. Defaults to top 5
                      contributors
  --limit INTEGER     Max number of issues to download. Defaults to 500
  --help              Show this message and exit.
```

Example usage, for the ngneat/elf repository:
```
% python ./src/main.py download \
  --data-dir ./data \
  --api-token $GITHUB_API_TOKEN \
  --users NetanelBasal,shaharkazaz,ido-golan,EricPoul \
  --project-url 'https://github.com/ngneat/elf' \
  --limit 10

Owner: nrwl, Repo: nx
Fetching issues, page: 1
Ignoring issue 20674. No relevant users in list ['bjornharvold']
Ignoring issue 20677. No relevant users in list ['e-oz']
Ignoring issue 20679. No relevant users in list ['karocksjoelee']
Saving issue 20676. User heike2718. Filepath ./issues/20676.txt
Ignoring issue 20678. No relevant users in list ['TriPSs', 'vercel[bot]', 'nx-cloud[bot]', 'TriPSs']
Ignoring issue 20671. No relevant users in list ['botre', 'TriPSs']
Ignoring issue 20673. No relevant users in list ['jluxenberg', 'andersonba']
Ignoring issue 20670. No relevant users in list ['barbados-clemens', 'vercel[bot]', 'nx-cloud[bot]']
Saving issue 20672. User eladhaim. Filepath ./issues/20672.txt
Ignoring issue 20675. No relevant users in list ['mklueh', 'mklueh']
Writing issue index to ./issues/index.json
```

Issues will be downloaded to the `./issues` folder, with the filename `./issues/<issue_number>.txt`.

Additionally, an index file will be created at `./issues/index.json`, which maps from users to the issues relevant to that user

Example output folder structure:
```
$DATA_DIR/
  - issues/
      - 20670.txt  # Full text and comments of issue 20670
      - 20672.txt
      - 20676.txt
      - 20678.txt
  - index.json  # Index of all issue files, maps each file to a relevant user
```

### Step 2 - Learn categories
The second step in classification is mapping documents into vector representations which we can compare.  In the learning step, we:
1. Tokenize all input files, and initialize bag-of-words vector dimensions
2. Create a term-frequency inverted-index for all files in the corpus
3. Calculate TF-IDF vectors for each document
4. Trial-run our classification algorithms, calculate accuracy for our training data-set

This learning step ONLY indexes 80% of the files in the corpus (selected randomly). The remaining 20% are used as an evaluation dataset.

To run the learning step, use the `learn` command
```
% python ./src/main.py learn
Usage: main.py learn [OPTIONS]

  Index 80% of the files in the corpus, and create tf-idf vector
  representations for all files.

  Run classification on 20% of the files, calculate accuracy.

Options:
  --data-dir TEXT  Folder to store issues, vectors
  --help           Show this message and exit.
```

Example usage:
```
% python ./src/main.py learn --data-dir ./data

Adding document to inverted index: ./data/issues/490.txt
Adding document to inverted index: ./data/issues/496.txt
Adding document to inverted index: ./data/issues/503.txt
Adding document to inverted index: ./data/issues/493.txt
Generating tf-idf for doc ./data/issues/490.txt
Generating tf-idf for doc ./data/issues/496.txt
Generating tf-idf for doc ./data/issues/503.txt
Generating tf-idf for doc ./data/issues/493.txt
-----------
Evaluation
-----------
./data/issues/487.txt - User NetanelBasal - Predicted User NetanelBasal
./data/issues/492.txt - User EricPoul - Predicted User NetanelBasal
-----------
Evaluation Accuracy
  NetanelBasal: 1 / 1 = 1.0
  EricPoul: 0 / 1 = 0.0
  overall: 1 / 2 = 0.5
```

The output files generated should look like:
```
$DATA_DIR/
  - documents/
    - $ISSUE_NUMBER_tf.txt      # Count of all terms in this issue
    - $ISSUE_NUMBER_tf_idf.txt  # Count of terms, with IDF weighting applied
    ...
    - $FILENAME.npy             # Numpy-compatible serialization of the .txt file with the same name
  - inverted_index.txt          # Term-Frequency inverted index
  - training.json               # Index of files used for training
  - evaluation.json             # Index of files used for evaluation
```

### Step 3 - Classify new documents
Now, the final step of classification is to:
1. Download a new github issue
2. Tokenize the issue, create a bag-of-words vector with the same dimensions as our training data
3. Use one of our classification algorithms to classify it into one of the learned categories
    - KNN
    - Naive Bayes
  
For this step, we have the `classify` command:
```
% python ./src/main.py classify
Usage: main.py classify [OPTIONS]

Options:
  --data-dir TEXT      Folder to store issues, vectors
  --method TEXT        Either "knn" or "naivebayes"
  --api-token TEXT     Github API token
  --github-issue TEXT  GitHub issue URL or issue number to classify
  --filepath TEXT      File to classify. If given, ignores github-issue and
                       api-token
  --verbose            If defined,
  --help               Show this message and exit.
```

Example usage:
```
% python ./src/main.py classify \
    --data-dir ./data \
    --method knn \
    --api-token $GITHUB_API_TOKEN \
    --github-issue  https://github.com/ngneat/elf/issues/503 \
    --verbose

The output should look like:

Filepath: ./data/issues/503.txt  Topic: NetanelBasal  Similarity: 1.0
Filepath: ./data/issues/496.txt  Topic: NetanelBasal  Similarity: 0.0915722322675805
Filepath: ./data/issues/493.txt  Topic: EricPoul  Similarity: 0.08406253459296453
Filepath: ./data/issues/490.txt  Topic: NetanelBasal  Similarity: 0.014407300373578137
Classification Results: NetanelBasal
Duplicate issue found: ./data/issues/503.txt 1.0
Duplicate docs:  ['./data/issues/503.txt']
```

## Team Member Contributions
| Main Tasks      | Estimated Time Cost (hours) |
| ----------- | ----------- |
| GitHub API code - Research and implement code to pull information from a GitHub repository (Yogi)   | 20 hours       |
| Tokenization, LM initialization - read project source code, issues to instantiate bag-of-words vector dimensions. Then transform issue descriptions into vectors. (Ben, Annamika, Shivani)|20 hours |
| User Profiles - Use git commit history and user-provided data to assign bag-of-words vectors to each maintainer of the project.  (Ben)    | 20 hours       |
| Similarity and Ranking - Implement ranking functions to generate a ranked list of maintainers for each Issue. (Annamika, Shivani, Ben) | 10 hours        |
| Integration and polish - Bring all these components together into a unified system that can actually be used by people.  (All)    | 10 hours       |
| (Stretch goal) Duplicate, similarity detection - use similarity between queries to automatically label duplicate or related issues. (Ben, Yogi)  | 10 hours        |
| **Total**      | **90 hours (10 stretch)**       |

# ByteCraft Documentation

## Overview of the Function of the Code
- An overview of the function of the code (i.e., what it does and what it can be used for).

## Implementation
- Documentation of how the software is implemented with sufficient detail so that others can have a basic understanding of your code for future extension or any further improvement. 

## Usage

### Setup and dependencies
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

Example usage, for the nrlw/nx repository:
```
python ./src/main.py download \
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
$data_dir/
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

### Step 3 - Classify new Documents
Now, the final step of classification is to:
1. Download a new github issue
2. Tokenize the issue, create a bag-of-words vector with the same dimensions as our training data
3. Use any classification algorithm to classify it into one of the learned categories
    - KNN
    - Naive Bayes
  
For this step, we have the `classify` command:
```
% python ./src/main.py classify
Usage: main.py classify [OPTIONS]

Options:
  --data-dir TEXT      Folder to store issues, vectors
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
    --api-token $GITHUB_API_TOKEN \
    --github-issue  https://github.com/ngneat/elf/issues/503 \
    --verbose

Euclidean distances: 
./data/issues/503.txt 0.0 - NetanelBasal
./data/issues/496.txt 22.148671156102935 - NetanelBasal
./data/issues/490.txt 24.302510198043095 - NetanelBasal
./data/issues/493.txt 27.928345601321087 - EricPoul

Classification Results: NetanelBasal
Duplicate issue found: ./data/issues/503.txt 1.0
Duplicate docs:  ['./data/issues/503.txt']
```

## Team Member Contributions
Ben Sivoravong (bs62): 
Shivani Mangaleswaran (sm131):
Annamika Dua (ad8):
Yogi Patel (ypatel55): 
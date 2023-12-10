# ByteCraft Documentation

## Overview of the Function of the Code
- An overview of the function of the code (i.e., what it does and what it can be used for).

## Implementation
- Documentation of how the software is implemented with sufficient detail so that others can have a basic understanding of your code for future extension or any further improvement. 

## Usage
### Step 1 - Download training data
The first step in the classification system is downloading historical issues from a Github repo, which becomes our training dataset. 

To download issues, generate a Github Personal Access Token following the instructions here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens

The `download` command will perform the folowing actions:
1. Find the top 5 contributors for a repo (if no users  are provided)
2. Look at the N most recent issues in the repo
3. Download any issues which are relevant to the given users
4. Create an `index.json` file, mapping issue txt files to their respective user (category)
```
% python main.py download
Usage: main.py download [OPTIONS]

Options:
  --api-token TEXT    Github API token
  --project-url TEXT  Full url to the project on github
  --users TEXT        Comma-separated list of users. Defaults to top 5
                      contributors
  --limit INTEGER     Max number of issues to download. Defaults to 500
  --help              Show this message and exit.
```

Example usage, for the nrlw/nx repository:
```
python main.py download \
--api-token <replaceme> \
--project-url 'https://github.com/nrwl/nx' \
--users 'eladhaim,heike2718' \
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
- main.py
- issues/
    - 20670.txt
    - 20672.txt
    - 20676.txt
    - 20678.txt
    - index.json
```

### Step 2 - Learn categories
The second step in classification is learning what each category looks like.  In the learning step, we:
1. Tokenize all input files, and initialize bag-of-words vector dimensions
2. Create a term-frequency inverted-index for all files in the corpus
3. Calculate maximum-likelihood estimator LM's for each document and category
3. Calculate TF-IDF vectors for each document and category

To run the learning step, use the `learn` command
```
% python main.py learn
Usage: main.py learn [OPTIONS]

  Do topic modeling on a set of input text files with known classifications.

  Estimates a ML Unigram LM for each topic given, and writes the LMs to an
  output directory.

Options:
  --index-file TEXT  Path to index file
  --output-dir TEXT  Folder to store output files
  --help             Show this message and exit.
```

Example usage:
```
% python main.py learn \
--index-file ./issues/index.json \
--output-dir ./output

Adding document to inverted index: ./issues/20676.txt
Adding document to inverted index: ./issues/20672.txt
Creating tf-vector, prob-dist for topic: heike2718
Creating tf-vector, prob-dist for topic: eladhaim
```

The output files generated should look like:
```
- main.py
- $output_dir/
    - categories/
        - $category/
            - tf.txt          # Count of all terms in this category's documents
            - tf_lm.txt       # Probability distribution generated from tf.txt
            - tf_idf.txt      # Count of all terms in this category, with IDF weighting applied
            - tf_idf_lm.txt   # Probability distribution generated from tf_idf.txt
            - $filename.npy   # numpy-compatible serialization of the .txt file with the same name
    - documents/
      - $filename_tf.txt      # Count of all terms in this document
      - $filename_tf_idf.txt  # Count of terms, with IDF weighting applied
      - $filename.npy         # numpy-compatible serialization of the .txt file with the same name
    - inverted_index.txt      # Term-Frequency inverted index used internally for learning
```

### Step 3 - Classify new Documents

Now, the final step of classification is to:
1. Read a new input file
2. Tokenize the input file, create a bag-of-words vector with the same dimensions as our training data
3. Use any classification algorithm to classify it into one of the learned categories
    - KNN
    - Naive Bayes
  
For this step, we have the `classify` command:
```
% python main.py classify
Usage: main.py classify [OPTIONS]

Options:
  --learn-dir TEXT  Folder generated from the "learn" step
  --filepath TEXT   File to classify
  --help            Show this message and exit.
```

Example usage:
```
% python main.py classify  \
--learn-dir ./output \
--filepath ./issues/20583.txt

Similarity with category eladhaim: 0.9325790152296436
Similarity with category heike2718: 0.7771994252477538
```

## Team Member Contributions
Ben Sivoravong (bs62): 
Shivani Mangaleswaran (sm131):
Annamika Dua (ad8):
Yogi Patel (ypatel55): 
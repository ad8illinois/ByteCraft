# Bytecraft - Github issue triage
Final presentation can be found here: [add video title and link]() 

## Team Members
Ben Sivoravong (bs62) 
Shivani Mangaleswaran (sm131)
Annamika Dua (ad8)
Yogi Patel (ypatel55):

## Usage
Complete usage directions can be found here: [documentation.md](https://github.com/ad8illinois/ByteCraft/blob/master/documentation.md) 

Create a python venv, install dependencies:
```
python -m venv venv
source venv/bin/activate

pip install -r ./requirements.txt
```

Install nltk data:
```
TODO
```


Then run the commands below:
```
GITHUB_API_TOKEN='replaceme'

# Step 1: Download issues from Github
python src/main.py download \
    --output-dir ./issues \
    --api-token $GITHUB_API_TOKEN \
    --users NetanelBasal,shaharkazaz,ido-golan,EricPoul \
    --project-url 'https://github.com/ngneat/elf'

# Step 2: Learn issue categories (users)
python src/main.py learn --index-file ./issues/index.json --output-dir ./output

# Step 3: Classify an issue from Github
python src/main.py classify \
    --learn-dir ./output \
    --api-token $GITHUB_API_TOKEN \
    --github-issue  https://github.com/ngneat/elf/issues/503
```

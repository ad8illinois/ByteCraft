# Github-Triage
A command-line utility for triage of github issues.

Final presentation can be found here: [add video title and link]() 

## Team Members
Ben Sivoravong (bs62) 
Shivani Mangaleswaran (sm131)
Annamika Dua (ad8)
Yogi Patel (ypatel55)

## Usage
Complete usage directions and details can be found here: [documentation.md](https://github.com/ad8illinois/ByteCraft/blob/master/documentation.md) 

### Docker
The easiest way to get started is to use the included docker container
```
docker build -t github-triage .

GITHUB_API_TOKEN='replaceme'

# Step 1: Download past issues
docker run -v $PWD/data:/data github-triage download \
    --output-dir /data/issues \
    --api-token $GITHUB_API_TOKEN \
    --users NetanelBasal,shaharkazaz,ido-golan,EricPoul \
    --project-url 'https://github.com/ngneat/elf'

# Step 2: Learn issue categories (users)
docker run -v $PWD/data:/data github-triage learn \
    --index-file /data/issues/index.json \
    --output-dir /data/output

# Step 3: Run classification on new issues
docker run -v $PWD/data:/data github-triage classify \
    --learn-dir /data/output \
    --api-token $GITHUB_API_TOKEN \
    --github-issue  https://github.com/ngneat/elf/issues/503
```

### Native (without Docker)
Create a python venv, install dependencies:
```
python -m venv venv
source venv/bin/activate

pip install -r ./requirements.txt
python -c "import nltk; nltk.download('popular')"
```

Then run the commands below:
```
GITHUB_API_TOKEN='replaceme'

# Step 1: Download issues from Github
python ./src/main.py download \
    --output-dir ./issues \
    --api-token $GITHUB_API_TOKEN \
    --users NetanelBasal,shaharkazaz,ido-golan,EricPoul \
    --project-url 'https://github.com/ngneat/elf'

# Step 2: Learn issue categories (users)
python src/main.py learn \
    --index-file ./issues/index.json \
    --output-dir ./output

# Step 3: Classify an issue from Github
python ./src/main.py classify \
    --learn-dir ./output \
    --api-token $GITHUB_API_TOKEN \
    --github-issue  https://github.com/ngneat/elf/issues/503
```

# Github-Triage
A command-line utility for triage of github issues.

Final presentation can be found here: [add video title and link]() 

## Team Members
Ben Sivoravong (bs62)\ 
Shivani Mangaleswaran (sm131)\
Annamika Dua (ad8)\
Yogi Patel (ypatel55)\

## Usage
Complete usage directions and details can be found here: [documentation.md](https://github.com/ad8illinois/ByteCraft/blob/master/documentation.md) 


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

python src/main.py download \
    --data-dir ./data \
    --api-token $GITHUB_API_TOKEN \
    --project-url 'https://github.com/keras-team/keras'

python src/main.py learn --data-dir ./data

python src/main.py classify \
    --data-dir ./data \
    --api-token $GITHUB_API_TOKEN \
    --github-issue https://github.com/keras-team/keras/issues/18943
```

### Docker
We also provide a dockerfile, so you can run these commands in docker.

```
GITHUB_API_TOKEN='replaceme'

docker build -t github-triage .

docker run -v $PWD/data:/data github-triage download \
    --data-dir /data \
    --api-token $GITHUB_API_TOKEN \
    --users NetanelBasal,shaharkazaz,ido-golan,EricPoul \
    --project-url 'https://github.com/ngneat/elf'

docker run -v $PWD/data:/data github-triage learn --data-dir /data

docker run -v $PWD/data:/data github-triage classify \
    --data-dir /data \
    --api-token $GITHUB_API_TOKEN \
    --github-issue  https://github.com/ngneat/elf/issues/503

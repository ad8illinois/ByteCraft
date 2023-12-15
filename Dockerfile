FROM python:3.9

WORKDIR /app

ADD requirements.txt ./

RUN pip install -r ./requirements.txt
RUN python -c "import nltk; nltk.download('popular')"

ADD ./src ./src

ENTRYPOINT ["python", "/app/src/main.py"]
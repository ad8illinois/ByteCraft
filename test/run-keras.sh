echo 'Github API Token:'
read GITHUB_API_TOKEN

python src/main.py download \
    --data-dir ./data \
    --api-token $GITHUB_API_TOKEN \
    --project-url 'https://github.com/keras-team/keras'

python src/main.py learn --data-dir ./data

python src/main.py classify \
    --data-dir ./data \
    --method knn \
    --api-token $GITHUB_API_TOKEN \
    --github-issue https://github.com/keras-team/keras/issues/18943

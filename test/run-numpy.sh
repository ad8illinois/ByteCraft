echo 'Github API Token:'
read GITHUB_API_TOKEN

python src/main.py download \
--output-dir ./issues \
--api-token $GITHUB_API_TOKEN \
--project-url 'https://github.com/numpy/numpy'

python src/main.py learn --index-file ./issues/index.json --output-dir ./output

python src/main.py classify \
    --learn-dir ./output \
    --api-token $GITHUB_API_TOKEN \
    --github-issue https://github.com/numpy/numpy/issues/25396
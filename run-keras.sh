# echo 'Github API Token:'
# read GITHUB_API_TOKEN

echo ""
echo "Downloading issues."
echo ""

python main.py download \
--output-dir ./issues \
--api-token $GITHUB_API_TOKEN \
--project-url 'https://github.com/keras-team/keras'

echo ""
echo "Downloading complete. Starting learning process."
echo ""

python main.py learn --index-file ./issues/index.json --output-dir output
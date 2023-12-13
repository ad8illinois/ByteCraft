echo 'Github API Token:'
read GITHUB_API_TOKEN

echo ""
echo "Downloading issues."
echo ""

python main.py download \
--output-dir ./issues \
--api-token $GITHUB_API_TOKEN \
--users NetanelBasal,shaharkazaz,ido-golan,EricPoul \
--project-url 'https://github.com/ngneat/elf'

echo ""
echo "Downloading complete. Starting learning process."
echo ""

python main.py learn --index-file ./issues/index.json --output-dir learned
echo ""
echo "Learning complete. Starting evaluation."
echo ""

echo "Random issues from NetanelBasal:"
python main.py classify --learn-dir ./learned --api-token $GITHUB_API_TOKEN --github-issue  https://github.com/ngneat/elf/issues/503
python main.py classify --learn-dir ./learned --api-token $GITHUB_API_TOKEN --github-issue  https://github.com/ngneat/elf/issues/316
python main.py classify --learn-dir ./learned --api-token $GITHUB_API_TOKEN --github-issue  https://github.com/ngneat/elf/issues/275
python main.py classify --learn-dir ./learned --api-token $GITHUB_API_TOKEN --github-issue  https://github.com/ngneat/elf/issues/34
echo ""
echo ""

echo "Random issues from EricPoul:"
python main.py classify --learn-dir ./learned --api-token $GITHUB_API_TOKEN --github-issue  https://github.com/ngneat/elf/issues/437
python main.py classify --learn-dir ./learned --api-token $GITHUB_API_TOKEN --github-issue  https://github.com/ngneat/elf/issues/363
python main.py classify --learn-dir ./learned --api-token $GITHUB_API_TOKEN --github-issue  https://github.com/ngneat/elf/issues/303
python main.py classify --learn-dir ./learned --api-token $GITHUB_API_TOKEN --github-issue  https://github.com/ngneat/elf/issues/276
echo ""
echo ""

echo "Random issues from shaharkazaz:"
python main.py classify --learn-dir ./learned --api-token $GITHUB_API_TOKEN --github-issue  https://github.com/ngneat/elf/issues/278
python main.py classify --learn-dir ./learned --api-token $GITHUB_API_TOKEN --github-issue  https://github.com/ngneat/elf/issues/242
python main.py classify --learn-dir ./learned --api-token $GITHUB_API_TOKEN --github-issue  https://github.com/ngneat/elf/issues/198
python main.py classify --learn-dir ./learned --api-token $GITHUB_API_TOKEN --github-issue  https://github.com/ngneat/elf/issues/6
echo ""
echo ""


echo "Random issues from ido-golan:"
python main.py classify --learn-dir ./learned --api-token $GITHUB_API_TOKEN --github-issue  https://github.com/ngneat/elf/issues/434
python main.py classify --learn-dir ./learned --api-token $GITHUB_API_TOKEN --github-issue  https://github.com/ngneat/elf/issues/8
python main.py classify --learn-dir ./learned --api-token $GITHUB_API_TOKEN --github-issue  https://github.com/ngneat/elf/issues/7
python main.py classify --learn-dir ./learned --api-token $GITHUB_API_TOKEN --github-issue  https://github.com/ngneat/elf/issues/2
echo ""
echo ""
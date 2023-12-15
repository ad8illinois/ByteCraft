echo 'Github API Token:'
read GITHUB_API_TOKEN

python main.py download \
--output-dir ./issues \
--api-token $GITHUB_API_TOKEN \
--users NetanelBasal,shaharkazaz,ido-golan,EricPoul \
--project-url 'https://github.com/ngneat/elf'

python main.py learn --index-file ./issues/index.json --output-dir ./output

python main.py classify --learn-dir ./output --api-token $GITHUB_API_TOKEN --github-issue  https://github.com/ngneat/elf/issues/503
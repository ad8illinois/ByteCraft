echo "Starting learning process."
echo ""

python ./src/main.py learn --index-file ./testdata/wikipedia/index.json --output-dir ./output
echo ""
echo "Learning complete. Starting evaluation."
echo ""

echo "Animals:"
python ./src/main.py classify --learn-dir ./output --filepath ./testdata/wikipedia/bird.txt
python ./src/main.py classify --learn-dir ./output --filepath ./testdata/wikipedia/cat.txt
python ./src/main.py classify --learn-dir ./output --filepath ./testdata/wikipedia/dog.txt
python ./src/main.py classify --learn-dir ./output --filepath ./testdata/wikipedia/fish.txt
python ./src/main.py classify --learn-dir ./output --filepath ./testdata/wikipedia/horse.txt
echo ""
echo ""

echo "Places"
python ./src/main.py classify --learn-dir ./output --filepath ./testdata/wikipedia/uiuc.txt
python ./src/main.py classify --learn-dir ./output --filepath ./testdata/wikipedia/chicago.txt
python ./src/main.py classify --learn-dir ./output --filepath ./testdata/wikipedia/dallas.txt
python ./src/main.py classify --learn-dir ./output --filepath ./testdata/wikipedia/champaign.txt
echo ""
echo ""

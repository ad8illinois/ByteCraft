echo "Starting learning process."
echo ""

python main.py learn --index-file ./testdata/wikipedia/index.json --output-dir learned
echo ""
echo "Learning complete. Starting evaluation."
echo ""

echo "Animals:"
python main.py classify --learn-dir ./learned --filepath ./testdata/wikipedia/bird.txt
python main.py classify --learn-dir ./learned --filepath ./testdata/wikipedia/cat.txt
python main.py classify --learn-dir ./learned --filepath ./testdata/wikipedia/dog.txt
python main.py classify --learn-dir ./learned --filepath ./testdata/wikipedia/fish.txt
python main.py classify --learn-dir ./learned --filepath ./testdata/wikipedia/horse.txt
echo ""
echo ""

echo "Places"
python main.py classify --learn-dir ./learned --filepath ./testdata/wikipedia/uiuc.txt
python main.py classify --learn-dir ./learned --filepath ./testdata/wikipedia/chicago.txt
python main.py classify --learn-dir ./learned --filepath ./testdata/wikipedia/dallas.txt
python main.py classify --learn-dir ./learned --filepath ./testdata/wikipedia/champaign.txt
echo ""
echo ""

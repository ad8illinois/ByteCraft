python ./src/main.py learn --data-dir ./testdata/wikipedia


echo "Animals:"
python ./src/main.py classify --data-dir ./testdata/wikipedia --method naivebayes --filepath ./testdata/wikipedia/pages/bird.txt
python ./src/main.py classify --data-dir ./testdata/wikipedia --method naivebayes --filepath ./testdata/wikipedia/pages/cat.txt
python ./src/main.py classify --data-dir ./testdata/wikipedia --method naivebayes --filepath ./testdata/wikipedia/pages/dog.txt
python ./src/main.py classify --data-dir ./testdata/wikipedia --method naivebayes --filepath ./testdata/wikipedia/pages/fish.txt
python ./src/main.py classify --data-dir ./testdata/wikipedia --method naivebayes --filepath ./testdata/wikipedia/pages/horse.txt
echo ""
echo ""

echo "Places"
python ./src/main.py classify --data-dir ./testdata/wikipedia --method naivebayes --filepath ./testdata/wikipedia/pages/uiuc.txt
python ./src/main.py classify --data-dir ./testdata/wikipedia --method naivebayes --filepath ./testdata/wikipedia/pages/chicago.txt
python ./src/main.py classify --data-dir ./testdata/wikipedia --method naivebayes --filepath ./testdata/wikipedia/pages/dallas.txt
python ./src/main.py classify --data-dir ./testdata/wikipedia --method naivebayes --filepath ./testdata/wikipedia/pages/champaign.txt
echo ""
echo ""

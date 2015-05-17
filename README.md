# Texture Matching using Local Binary Patterns (LBP)

Related blog post explaining the code - [Click
Here](http://hanzratech.in/2015/05/30/local-binary-patterns.html)

## Usage

__Perform training using__

```
python perform-training.py -t data/lbp/train/ -l data/lbp/class_train.txt
```

__Perform testing using__

```
python perform-testing.py -t data/lbp/test/ -l data/lbp/class_test.txt
```
## Results

Let's check out the results.

__Input Image -- ROCK Class__

Here is an input image of a rock texture.

![alt tag](docs/images/query-image-1.png)

_Matching Scores_ - Below are the sorted results of matching. The top 2 scores are of rock texture.

![alt tag](docs/images/query-image-results-1.png)

__Input Image -- CHECKERED Class__

Here is an input image of a checkered texture .

![alt tag](docs/images/query-image-2.png)

_Matching Scores_ - Below are the sorted results of matching. The top 2 scores are of checkered texture.

![alt tag](docs/images/query-image-results-2.png)

__Input Image -- GRASS Class__

Here is an input image of a grass texture.

![alt tag](docs/images/query-image-3.png)

_Matching Scores_ - Below are the sorted results of matching. The top 2 scores are of grass texture.

![alt tag](docs/images/query-image-results-3.png)

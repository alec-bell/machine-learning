# Batch Perceptron

`perceptron.py` implements the Batch Perceptron algorithm. I trained and tested it on the Iris dataset, and simplified it by only evaluating 2 features to make the data linearly separable. It only checks whether or not a particular example should be classified as a satosa or not. Finally, I added a feature to the train and test examples to make the data linearly separable through the origin (no need for an offset).

## Analysis

After running the algorithm on the training data, it took a total of `12` iterations to converge.

This resulted in an accuracy of `98.91%` on the test set, and a geometric margin of `3.24`.
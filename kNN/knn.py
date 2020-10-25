import numpy as np
import matplotlib.pylab as plt
%matplotlib inline

# Simple wrapper class for the data we'll use to train the following kNN implementation
class Numbers:
    """
    Class to store MNIST data
    """
    def __init__(self, location):

        import pickle, gzip

        # load data from file 
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = pickle.load(f)
        f.close()

        # store for use later  
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set

DATA_PATH = "data/mnist.pklz"
data = Numbers(DATA_PATH)

class Knearest:
    """
    kNN classifier
    """

    def __init__(self, X, y, k=5, p=10):
        """
        Creates a kNN instance

        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        :param p: The number of prototypes to choose
        """
        from sklearn.neighbors import BallTree
        self._p = p
        self._x, self._y = self.select_prototypes(X, y)
        self._k = k
        self._counts = self.label_counts(y)
        self._kdtree = BallTree(self._x)
            
    def label_counts(self, y):
        """
        Given the training prototypes, return a dictionary d where d[y] is  
        the number of times that label y appears in the training prototypes.
        
        :param y: Training labels input
        """  
        d = dict()
        for label in y:
            if label in d:
                d[label] += 1
            else:
                d[label] = 1
        return d


    def majority(self, neighbor_indices):
        """
        Given the indices of training protypes, return the majority label. Break ties 
        by choosing the tied label that appears most often in the training prototypes. 

        :param neighbor_indices: The indices of the k nearest neighbors
        """
        assert len(neighbor_indices) == self._k, "Did not get k neighbor indices"
        
        y = self._y[neighbor_indices]
        label_counts = self.label_counts(y)
        
        majority_labels = list()
        max = -1
        for label in label_counts:
            if label_counts[label] > max:
                majority_labels = list()
                majority_labels.append(label)
                max = label_counts[label]
            elif label_counts[label] == max:
                majority_labels.append(label)
        
        # if there's only one maximum, then return it
        if len(majority_labels) == 1:
            return majority_labels[0]
        
        # otherwise, return the maximum that appears the most in the training set
        max_label = majority_labels[0]
        max = self._counts[majority_labels[0]]
        for label in self._counts:
            if label in majority_labels and self._counts[label] > max:
                max_label = label
                max = self._counts[label]
        return max_label
        
    def classify(self, example):
        """
        Given an example, return the predicted label.

        :param example: A representation of an example in the same
        format as a row of the training data
        """
        ex = np.array([example])
        neighbor_indices = self._kdtree.query(ex, k=self._k, return_distance=False)
        return int(self.majority(neighbor_indices[0]))
        
        
    def select_prototypes(self, X, y):
        """
        Given the train data, select p random prototypes from data for classification
        calculation.
        
        :param x: Training data input
        :param y: Training data output
        """
        import random
        
        idx = np.random.choice(np.arange(len(X)), self._p, replace=False)
        X_prototypes = X[idx]
        y_prototypes = y[idx]
        return X_prototypes, y_prototypes
            
    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrix for the current classifier.  Should return a 2-dimensional
        numpy array of ints, C, where C[ii,jj] is the number of times an 
        example with true label ii was labeled as jj.

        :param test_x: test data 
        :param test_y: true test labels 
        """
        C = np.zeros((10,10), dtype=int)
        for xx, yy in zip(test_x, test_y):
            prediction = self.classify(xx)
            C[yy][prediction] += 1
        return C
            
    @staticmethod
    def accuracy(C):
        """
        Given a confusion matrix C, compute the accuracy of the underlying classifier.
        
        :param C: a confusion matrix 
        """
        return np.sum(C.diagonal()) / C.sum()

# The code below creates a plot exploring the relationship between the value of k (number of nearest neighbors)
# and the accuracy on the test set.

x,y=list(),list()
for k in range(1, 20):
    sum_accuracy = 0
    for i in range(10):
        knearest = Knearest(data.train_x, data.train_y, k=k, p=1000)
        accuracy = Knearest.accuracy(knearest.confusion_matrix(data.test_x, data.test_y))
        sum_accuracy += accuracy
    avg_accuracy = sum_accuracy / 10
    x.append(k)
    y.append(avg_accuracy)
plt.plot(x, y)
plt.xlabel('# of nearest neighbors', fontsize=12)
plt.ylabel('avg accuracy after 10 attempts', fontsize=12)
plt.show()

# The code below creates a confusion matrix of predicted label vs. actual label for each digit,
# and also shows us examples of the digits that were misclassified.

def image_matrix(knearest, test_x, test_y):
    images = np.zeros((10, 10, 784))
    for xx, yy in zip(test_x, test_y):
        prediction = knearest.classify(xx)
        images[yy][prediction] = xx
    return images

import seaborn as sn
import pandas as pd

knearest = Knearest(data.train_x, data.train_y, k=1, p=1000)
confusion_matrix = knearest.confusion_matrix(data.test_x, data.test_y)
image_matrix = image_matrix(knearest, data.train_x, data.train_y)
df_cm = pd.DataFrame(confusion_matrix, range(10), range(10))
plt.figure(figsize=(10,10))
sn.set(font_scale=1)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}) # found this cool heatmap library
plt.xlabel('true label', fontsize=12)
plt.ylabel('predicted label', fontsize=12)
plt.show()

# The code below allows us to look at handwritten digits that were misclassified by
# our kNN implementation.

def view_digit(example, label=None, prediction_label=None):
    if label: print("true label: {:d}".format(label))
    if prediction_label: print("predicted label: {:d}".format(prediction_label))
    plt.imshow(example.reshape(28,28), cmap='gray')

view_digit(image_matrix[9][4], 9, 4) # 9 was frequently misclassifed as 4
view_digit(image_matrix[4][9], 4, 9) # 4 was frequently misclassifed as 9
view_digit(image_matrix[7][9], 7, 9) # 7 was sometimes misclassified as 9
view_digit(image_matrix[5][3], 5, 3) # 5 was sometimes misclassified as 3
view_digit(image_matrix[3][5], 3, 5) # 3 was sometimes misclassified as 5

# The code below explores the relationship between the number of prototypes
# (the number of examples we sample from the training set) and accuracy.

x,y=list(),list()
for p in range(1000, 20001, 1000):
    knearest = Knearest(data.train_x, data.train_y, k=1, p=p)
    accuracy = Knearest.accuracy(knearest.confusion_matrix(data.test_x, data.test_y))
    x.append(p)
    y.append(accuracy)
plt.plot(x, y)
plt.xlabel('# of prototypes', fontsize=12)
plt.ylabel('accuracy', fontsize=12)
plt.show()
import numpy as np
import matplotlib.pylab as plt
%matplotlib inline

# Simple wrapper class for the data we'll use to train the following perceptron implementation
class IrisM:
    """
    Class to store modified iris data for Perceptron Training
    """
    
    def __init__(self):
        from sklearn import datasets
        
        iris = datasets.load_iris()
        
        # only taking first two features
        X = iris.data[:, :2]
        y = iris.target[:]
        
        # only considering whether it is setosa or not
        y[iris.target != 0] = -1
        y[iris.target == 0] = 1
        
        mask = np.random.choice(a = [False, True], size = 150, p = (0.66, 1 - 0.66))
        
        self.train_x, self.train_y = X[mask], y[mask]
        self.test_x, self.test_y = X[~mask], y[~mask]
        
iris = IrisM()

class Perceptron:
    """
    Perceptron Classifier
    """
    
    def __init__(self, X, y):
        """
        Creates a kNN instance

        :param X: Training data input
        :param y: Training data output
        """
        self._X = X
        self._y = y
        self._theta, self._iter = self.train(X, y)
        
    def train(self, X, y):
        """
        Train perceptron and return final classification vector and
        the number of updates performed respectively
        
        :param X: Training data input
        :param y: Training data output
        """
        num_features = X[0].size
        theta = np.zeros(num_features)
        did_misclassify = True
        iters = 0
        while did_misclassify: # do until convergence
            did_misclassify = False
            for i in range(len(X)):
                if np.dot(X[i], theta)*y[i] <= 0:
                    theta = theta + X[i]*y[i]
                    did_misclassify = True
            iters += 1
        return theta, iters
        
    
    def predict(self, X):
        """
        Predicts the label for input
        
        :param X: Testing data input
        """
        return 1 if np.dot(X, self._theta) > 0 else -1
        
    def margin(self):
        """
        Returns geometric margin of the classifier
        """
        import sys
        min_margin = sys.float_info.max
        transposed = np.transpose(self._theta)
        for x,y in zip(self._X, self._y):
            dist = np.linalg.norm(transposed*x/np.linalg.norm(self._theta))
            min_margin = min(min_margin, dist)
        return min_margin
    
# The code below adds a feature to the training and test examples so we can
# have a classifier that linearly separates the data through the origin.

num_features = 2
def add_feature(arr):
    num_examples = len(arr)
    new_arr = np.zeros((num_examples, num_features+1))
    for i in range(num_examples):
        new_arr[i][0] = arr[i][0]
        new_arr[i][1] = arr[i][1]
        new_arr[i][2] = 1
    return new_arr
        
train_x = add_feature(iris.train_x)
train_y = iris.train_y
test_x = add_feature(iris.test_x)
test_y = iris.test_y

# Now we train the perceptron

perceptron = Perceptron(train_x, train_y)

# The code below prints out the accuracy on the test set, the number of iterations
# it took for the perceptron algorithm to converge, and the geometric margin
# of the classifier (distance between parameter vector and the closest point to it).

errors = 0
for x,y in zip(test_x, test_y):
    prediction = perceptron.predict(x)
    if prediction != y:
        errors += 1
accuracy = (len(test_x) - errors) / len(test_x)

print("number of iterations to converge:\t" + str(perceptron._iter))
print("accuracy:\t\t\t\t" + str(accuracy))
print("geometric margin: " + str(perceptron.margin()))
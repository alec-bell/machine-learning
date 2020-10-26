import numpy as np
from sklearn.base import clone

# Wrapper class for data
class ThreesAndEights:
    """
    Class to store MNIST data
    """

    def __init__(self, location):

        import pickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')

        # Split the data set 
#         X_train, y_train, X_valid, y_valid = pickle.load(f)
        train_set, valid_set, test_set = pickle.load(f)
    
        X_train, y_train = train_set
        X_valid, y_valid = valid_set

        # Extract only 3's and 8's for training set 
        self.X_train = X_train[np.logical_or( y_train==3, y_train == 8), :]
        self.y_train = y_train[np.logical_or( y_train==3, y_train == 8)]
        self.y_train = np.array([1 if y == 8 else -1 for y in self.y_train])
        
        # Shuffle the training data 
        shuff = np.arange(self.X_train.shape[0])
        np.random.shuffle(shuff)
        self.X_train = self.X_train[shuff,:]
        self.y_train = self.y_train[shuff]

        # Extract only 3's and 8's for validation set 
        self.X_valid = X_valid[np.logical_or( y_valid==3, y_valid == 8), :]
        self.y_valid = y_valid[np.logical_or( y_valid==3, y_valid == 8)]
        self.y_valid = np.array([1 if y == 8 else -1 for y in self.y_valid])
        
        f.close()

data = ThreesAndEights("data/mnist.pklz")

# Import the RandomDecisionTree implementation from the random-decision-tree project in this repository

from "../random-decision-tree" import "random-decision-tree.py"
from "random-decision-tree.py" import RandomDecisionTree

class RandomDecisionForest:
    def __init__(self, ratio = 0.63, N = 20, max_depth = 10, candidate_splits = 500):
        """
        Create a new RandomDecisionForest
        
        Args:
            base (BaseEstimator, optional): Sklearn implementation of decision tree
            ratio: ratio of number of data points in subsampled data to the actual training data
            N: number of base estimator in the ensemble
        
        Attributes:
            base (estimator): Sklearn implementation of decision tree
            N: Number of decision trees
            learners: List of models trained on bootstrapped data sample
        """
        assert ratio <= 1.0, "Cannot have ratio greater than one"
        self.ratio = ratio
        self.N = N  
        self.learners = []
        self.candidate_splits = candidate_splits
        self.max_depth = max_depth
        
    def fit(self, X_train, y_train):
        """
        Train RandomDecisionForest Classifier on data
        
        Args:
            X_train (ndarray): [n_samples x n_features] ndarray of training data   
            y_train (ndarray): [n_samples] ndarray of data 
        """
        for i in range(self.N):
            h = RandomDecisionTree(candidate_splits=self.candidate_splits, depth=self.max_depth)
            h = h.fit(*self.bootstrap(X_train, y_train))
            self.learners.append(h)
        
    def bootstrap(self, X_train, y_train):
        """
        Args:
            n (int): total size of the training data
            X_train (ndarray): [n_samples x n_features] ndarray of training data   
            y_train (ndarray): [n_samples] ndarray of data 
        """
        size = int(len(X_train)*self.ratio)
        idx = np.random.choice(np.arange(len(X_train)), size, replace=True)
        X_train_subset = X_train[idx]
        y_train_subset = y_train[idx]
        return X_train_subset, y_train_subset
    
    def predict(self, X):
        """
        BaggingClassifier prediction for data points in X
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            
        Returns:
            yhat (ndarray): [n_samples] ndarray of predicted labels {-1,1}
        """
        y_hats = np.zeros(self.N)
        for i in range(self.N):
            y_hats[i] = self.learners[i].predict(X)
        return y_hats
    
    def voting(self, y_hats):
        """
        Args:
            y_hats (ndarray): [N] ndarray of data
        Returns:
            y_final : int, final prediction of the 
        """
        import random
        votes = dict()
        for y_hat in y_hats:
            if y_hat in votes:
                votes[y_hat] += 1
            else:
                votes[y_hat] = 1
        winners = set()
        max = 0
        for y_hat, counts in votes.items():
            if counts > max:
                winners = set()
                winners.add(y_hat)
                max = counts
            elif counts == max:
                winners.add(y_hat)
        return random.choice(tuple(winners))

# The code below uses K-Fold Cross Validation to evaluate the performance of the RandomDecisionForest in classifying
# 3's and 8's from the MNIST dataset.

K = 3 # Let K be 3 just for the sake of not getting too computationally complex

kf = KFold(n_splits=K)
rf = RandomForest()

depths = np.arange(1,11,3)
N_values = np.arange(1,26,6)
depths, N_values = np.meshgrid(depths, N_values, indexing='ij')
accuracies = np.zeros(depths.shape)
for i in range(len(depths)):
    for j in range(len(N_values[i])):
        rf = RandomForest(N=N_values[i,j], max_depth=depths[i,j])
        sum_accuracy_across_folds = 0 
        for train_index, test_index in kf.split(data.X_train):
            X_train, X_test = data.X_train[train_index], data.X_train[test_index]
            y_train, y_test = data.y_train[train_index], data.y_train[test_index]
            rf.fit(X_train, y_train)
            sum_accuracy_across_folds += accuracy(rf, X_test, y_test)
        accuracies[i,j] = sum_accuracy_across_folds / K

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(depths, N_values, accuracies, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('max depth')
ax.set_ylabel('# of random decision trees (N)')
ax.set_zlabel('avg accuracy over 4 folds')
plt.show()

# The code below finds the optimal values for the hyperparameters corresponding to the maximum allowed depth of the
# RandomDecisionTree's (max depth) and the number of RandomDecisionTree's (N) to include in the forest.

max_accuracy = 0
best_max_depth = 0
best_N = 0
for i in range(len(depths)):
    for j in range(len(N_values[i])):
        current_accuracy = accuracies[i,j]
        if (current_accuracy > max_accuracy):
            max_accuracy = current_accuracy
            best_max_depth = i
            best_N = j
print("Max avg accuracy over 3 folds:\t" + str(max_accuracy))
print("Optimal max depth:\t\t" + str(best_depth))
print("Optimal N:\t\t\t" + str(best_N))

# After finding the optimal values of max_depth and N through K-Fold Cross Validation, I then evaluated the performance 
# on the test set

rf = RandomForest(N=4, max_depth=7)
rf.fit(data.X_train, data.y_train)
validation_accuracy = accuracy(bagging_classifier, data.X_valid, data.y_valid)
print("Accuracy on validation set using optimal values:\t" + str(validation_accuracy))
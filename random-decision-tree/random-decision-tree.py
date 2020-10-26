import numpy as np
from sklearn.base import clone

# We are implementing this as a binary classifier, hence we will only use 3's and 8's from the training set.
# The numbers 3 and 8 are very similar in appearance so I chose them to increase the difficult of classification.

class ThreesAndEights:
    """
    Class to store MNIST data
    """
    def __init__(self, location):

        import pickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')

        # Split the data set 
        # X_train, y_train, X_valid, y_valid = pickle.load(f)
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

# Helper class just for organization of the Random Decision Tree
class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.isLeaf = False
        self.label = None
        self.split_vector = None

    def getLabel(self):
        if not self.isLeaf:
            raise Exception("Should not do getLabel on a non-leaf node")
        return self.label
    
class RandomDecisionTree:
            
    def __init__(self, candidate_splits = 100, depth = 10):
        """
        Args:
            candidate_splits (int) : number of random decision splits to test
            depth (int) : maximum depth of the random decision tree
        """
        self.candidate_splits = candidate_splits
        self.depth = depth
        self.root = None
    
    def fit(self, X_train, y_train):
        """
        Args:
            X_train (ndarray): [n_samples x n_features] ndarray of training data   
            y_train (ndarray): [n_samples] ndarray of data
            
        """
        self.root = self.build_tree(X_train[:], y_train[:], 0)
        return self
        
    def build_tree(self, X_train, y_train, height):
        """
        Args:
            X_train (ndarray): [n_samples x n_features] ndarray of training data   
            y_train (ndarray): [n_samples] ndarray of data
            
       Return:
            split_vector: random vector which gives most reduction in uncertainty
            feature_indices: indices of the random sub-features used
            lindices: indices of training example which should be in left subtree
            rindices: indices of training example which should be in right subtree
            
        """
        def build(node, X_train, y_train, height):
            if self.gini_index(y_train) == 0 or height == self.depth:
                node.isLeaf = True
                node.label = self.majority(y_train)
            else:
                node.split_vector = self.find_best_split(X_train, y_train)
                lindices, rindices = list(), list()
                for i in range(len(X_train)):
                    if np.dot(node.split_vector, X_train[i]) <= 0:
                        lindices.append(i)
                    else:
                        rindices.append(i)
                if len(lindices) == 0:
                    node.isLeaf = True
                    node.label = self.majority(y_train[rindices])
                elif len(rindices) == 0:
                    node.isLeaf = True
                    node.label = self.majority(y_train[lindices])
                else:
                    node.left = TreeNode()
                    node.right = TreeNode()
                    build(node.left, X_train[lindices], y_train[lindices], height+1)
                    build(node.right, X_train[rindices], y_train[rindices], height+1)
        node = TreeNode()
        build(node, X_train, y_train, 1)
        return node
        
    
    def find_best_split(self, X_train, y_train):
        """
        Args:
            X_train (ndarray): [n_samples x n_features] ndarray of training data   
            y_train (ndarray): [n_samples] ndarray of data
            
        """
        import random
        min_impurity = 100000000 # just any large number
        best_split_vector = None
        for i in range(self.candidate_splits):
            # The logic here: Picks a random sample from X_train and 3 random features from that sample.
            # Then creates a split vector full of zeros but populates the matching indices of the 3
            # random features with the 3 random features.
            # I chose "3" as the magic number for the number of random features because that seemed
            # to maximize my accuracy across runs.
            split_vector = np.zeros(X_train[0].shape)
            feature_idx = np.random.randint(len(X_train[0]), size=3)
            sample_idx = np.random.randint(len(X_train), size=1)
            split_vector[feature_idx] = X_train[sample_idx, feature_idx]
            
            left_y_train, right_y_train = list(), list()
            for X,y in zip(X_train, y_train):
                if np.dot(split_vector, X) <= 0:
                    left_y_train.append(y)
                else:
                    right_y_train.append(y)
            # impurity here is treating the gini_index of each side of the split in a weighted manner by multiplying it by the number
            # of elements in each
            impurity = len(left_y_train)*self.gini_index(left_y_train) + len(right_y_train)*self.gini_index(right_y_train)
            if impurity < min_impurity:
                min_impurity = impurity
                best_split_vector = split_vector
        return best_split_vector
        
    def gini_index(self, y):
        """
        Args:
            y (ndarray): [n_samples] ndarray of data
        """
        label_counts = dict()
        for label in y:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        gini = 0
        for label, count in label_counts.items():
            p = count / len(y)
            not_p = 1 - p
            gini += p * not_p
        return gini
        
    
    def majority(self, y):
        """
        Return the major class in ndarray y
        """
        import random
        label_counts = dict()
        for label in y:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        winners = set()
        max = 0
        for label, count in label_counts.items():
            if count > max:
                winners = set()
                winners.add(label)
                max = count
            elif count == max:
                winners.add(label)
        return random.choice(tuple(winners))
    
    def predict(self, X):
        """
        BaggingClassifier prediction for new data points in X
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            
        Returns:
            yhat (ndarray): [n_samples] ndarray of predicted labels {-1,1}
        """
        def traverse(node, X):
            if node.isLeaf:
                return node.label
            elif np.dot(node.split_vector, X) <= 0:
                return traverse(node.left, X)
            else:
                return traverse(node.right, X)
        
        return traverse(self.root, X)

# The code below simply evaluates the accuracy of this Random Decision Tree trained and tested on 3's and 8's from the MNIST data.

def rdt_accuracy(rdt, X_valid, y_valid):
    num_correct = 0
    for X,y in zip(X_valid, y_valid):
        prediction = rdt.predict(X)
        if prediction == y:
            num_correct += 1
    acc = num_correct / len(X_valid)
    return acc

rdt = RandomDecisionTree(depth=10)
rdt = rdt.fit(data.X_train, data.y_train)
acc = rdt_accuracy(rdt, data.X_valid, data.y_valid)
print("accuracy:\t" + str(acc))

# The code below creates a plot to evaluate the relationship between accuracy and the maximum allowed depth (a hyperparameter) 
# of the Random Decision Tree.

depths = np.arange(1, 21)
accuracies = np.zeros((20,))
for i in range(len(depths)):
    rdt = RandomDecisionTree(depth=depths[i])
    rdt.fit(data.X_train, data.y_train)
    accuracies[i] = rdt_accuracy(rdt, data.X_valid, data.y_valid)
plt.plot(depths, accuracies)
plt.xlabel('max depth', fontsize=12)
plt.ylabel('accuracy', fontsize=12)
plt.show()
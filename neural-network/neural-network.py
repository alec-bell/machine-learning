import numpy as np
import matplotlib.pyplot as plt

class Network:
    def __init__(self, sizes):
        """
        Initialize the neural network 
        
        :param sizes: a list of the number of neurons in each layer 
        """
        # save the number of layers in the network 
        self.L = len(sizes) 
        
        # store the list of layer sizes 
        self.sizes = sizes  
        
        # initialize the bias vectors for each hidden and output layer 
        self.b = [np.random.randn(n) for n in self.sizes[1:]]
        
        # initialize the matrices of weights for each hidden and output layer 
        self.W = [np.random.randn(n, m) for (m,n) in zip(self.sizes[:-1], self.sizes[1:])]
        
        # initialize the derivatives of biases for backprop 
        self.db = [np.zeros(n) for n in self.sizes[1:]]
        
        # initialize the derivatives of weights for backprop 
        self.dW = [np.zeros((n, m)) for (m,n) in zip(self.sizes[:-1], self.sizes[1:])]
        
        # initialize the activities on each hidden and output layer 
        self.z = [np.zeros(n) for n in self.sizes]
        
        # initialize the activations on each hidden and output layer 
        self.a = [np.zeros(n) for n in self.sizes]
        
        # initialize the deltas on each hidden and output layer 
        self.delta = [np.zeros(n) for n in self.sizes]
        
        self.train_accuracy = []
        
        self.valid_accuracy = []
        
        self.epochs = []
        
    def g(self, z):
        """
        sigmoid activation function 
        
        :param z: vector of activities to apply activation to 
        """
        z = np.clip(z, -20, 20)
        return 1.0/(1.0 + np.exp(-z))
    
    def g_prime(self, z):
        """
        derivative of sigmoid activation function 
        
        :param z: vector of activities to apply derivative of activation to 
        """
        return self.g(z) * (1.0 - self.g(z))
    
    def gradC(self, a, y):
        """
        evaluate gradient of cost function for squared-loss C(a,y) = (a-y)^2/2 
        
        :param a: activations on output layer 
        :param y: vector-encoded label 
        """
        return (a - y)
    
    def forward_prop(self, x):
        """
        take an feature vector and propagate through network 
        
        :param x: input feature vector 
        """
        self.a[0] = x
        for i in range(self.L-1):
            self.z[i+1] = np.dot(self.W[i], self.a[i]) + self.b[i]
            self.a[i+1] = self.g(self.z[i+1])
        
    def predict(self, X):
        """
        Predicts on the the data in X. Assume at least two output neurons so predictions
        are one-hot encoded vectorized labels. 
        
        :param X: a matrix of data to make predictions on 
        :return y: a matrix of vectorized labels 
        """
        yhat = np.zeros((X.shape[0], self.sizes[-1]), dtype=int)
        for i in range(len(X)):
            self.forward_prop(X[i])
            max_idx = np.argmax(self.a[self.L-1])
            yhat[i][max_idx] = 1
        return yhat
    
    def accuracy(self, X, y):
        """
        compute accuracy on labeled training set 

        :param X: matrix of features 
        :param y: matrix of vectorized true labels 
        """
        yhat = self.predict(X)
        return np.sum(np.all(np.equal(yhat, y), axis=1)) / X.shape[0]
            
    def back_prop(self, x, y):
        """
        Back propagation to get derivatives of C wrt weights and biases for given training example
        
        :param x: training features  
        :param y: vector-encoded label 
        """
        self.forward_prop(x)
        
        gradC = self.gradC(self.a[-1], y)
        derZ = self.g_prime(self.z[-1])
        self.delta[-1] = np.multiply(gradC, derZ)
        
        for ll in range(self.L-2, -1, -1):
            self.dW[ll] = np.outer(self.delta[ll+1], self.a[ll])
            self.db[ll] = self.delta[ll+1]
            W_dot_d = np.dot(self.W[ll].transpose(), self.delta[ll+1])
            derZ = self.g_prime(self.z[ll])
            self.delta[ll] = np.multiply(W_dot_d, derZ)
            
    def train(self, X_train, y_train, X_valid=None, y_valid=None, eta=0.25, lam=0.0, num_epochs=10, isPrint=True, shouldStore=False):
        """
        Train the network with SGD 
        
        :param X_train: matrix of training features 
        :param y_train: matrix of vector-encoded training labels 
        :param X_train: optional matrix of validation features 
        :param y_train: optional matrix of vector-encoded validation labels 
        :param eta: learning rate 
        :param lam: regularization strength 
        :param num_epochs: number of epochs to run 
        :param isPrint: flag indicating to print training progress or not 
        """
        
        # reset arrays storing accuracy
        self.epochs, self.train_accuracy, self.valid_accuracy = [], [], []
        
        # initialize shuffled indices 
        shuffled_inds = list(range(X_train.shape[0]))
        
        # loop over training epochs 
        for ep in range(num_epochs):
            
            # shuffle indices 
            np.random.shuffle(shuffled_inds)
            
            # loop over training examples 
            for ind in shuffled_inds:
                
                # TODO: back prop to get derivatives 
                self.back_prop(X_train[ind], y_train[ind])
                
                # TODO: update weights and biases 
                for i in range(len(self.W)):
                    for j in range(len(self.W[i])):
                        self.W[i][j] = self.W[i][j] - eta*self.dW[i][j] - eta*lam*self.W[i][j]
                
                for i in range(len(self.b)):
                    self.b[i] = self.b[i] - eta*self.db[i]
                
            # occasionally print accuracy
            if isPrint and ((ep+1)%10)==1:
                self.epoch_report(ep, num_epochs, X_train, y_train, X_valid, y_valid)
            elif shouldStore and ((ep+1)%5)==1:
                self.epoch_store_accuracies(ep, X_train, y_train, X_valid, y_valid)
                
        # print final accuracy
        if isPrint:
            self.epoch_report(ep, num_epochs, X_train, y_train, X_valid, y_valid)
                
                    
    def epoch_report(self, ep, num_epochs, X_train, y_train, X_valid, y_valid):
        """
        Print the accuracy for the given epoch on training and validation data 
        
        :param ep: the current epoch 
        :param num_epochs: the total number of epochs
        :param X_train: matrix of training features 
        :param y_train: matrix of vector-encoded training labels 
        :param X_train: optional matrix of validation features 
        :param y_train: optional matrix of vector-encoded validation labels 
        """
        
        print("epoch {:3d}/{:3d}: ".format(ep+1, num_epochs), end="")
        print("  train acc: {:8.3f}".format(self.accuracy(X_train, y_train)), end="")
        if X_valid is not None: print("  valid acc: {:8.3f}".format(self.accuracy(X_valid, y_valid)))
        else: print("")   
            
    def epoch_store_accuracies(self, ep, X_train, y_train, X_valid, y_valid):
        self.epochs.append(ep+1)
        self.train_accuracy.append(self.accuracy(X_train, y_train))
        self.valid_accuracy.append(self.accuracy(X_valid, y_valid))
    
    def plot_validation_accuracy(self):
        plt.plot(self.epochs, self.valid_accuracy)
        plt.xlabel('# of epochs', fontsize=12)
        plt.ylabel('validation accuracy', fontsize=12)
        plt.show()
    
# Import handwritten digit image data for testing the Network class
import pickle
import gzip
X_train, y_train, X_valid, y_valid = pickle.load(gzip.open("data/mnist21x21_3789_one_hot.pklz", "rb"))

# After conducting some tests, I found a learning rate of 1.5 to be good,
# as well as a regularization parameter of 0.000001.
# Here are a number of different architectures using the parameters above.

# One hidden layer of width 100
sizes = np.array([441, 100, 4])
nn = Network(sizes=sizes)
nn.train(X_train, y_train, X_valid, y_valid, eta=1.5, lam=0.000001, num_epochs=50, isPrint=False, shouldStore=True)
nn.plot_validation_accuracy()

# One hidden layer of width 150
sizes = np.array([441, 150, 4])
nn = Network(sizes=sizes)
nn.train(X_train, y_train, X_valid, y_valid, eta=1.5, lam=0.000001, num_epochs=50, isPrint=False, shouldStore=True)
nn.plot_validation_accuracy()

# Two hidden layers of width 200 and 100 respectively
sizes = np.array([441, 200, 100, 4])
nn = Network(sizes=sizes)
nn.train(X_train, y_train, X_valid, y_valid, eta=1.5, lam=0.000001, num_epochs=50, isPrint=False, shouldStore=True)
nn.plot_validation_accuracy()
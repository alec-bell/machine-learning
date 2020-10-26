import numpy as np
from scipy.io import loadmat

# Wrapper for preprocessed medical data on cardiac Single Proton Emission Tomography
# images of patients labeling them as either normal (1) or abnormal (0).
class SPECT:
    def __init__(self):
        ff = lambda x, y : loadmat(x)[y]
        
        self.X_train = ff('data/SPECTtrainData.mat','trainData')
        self.y_train = ff('data/SPECTtrainLabels.mat','trainLabels')
        
        self.X_test = ff('data/SPECTtestData.mat', 'testData')
        self.y_test = ff('data/SPECTtestLabels.mat', 'testLabels')       

# Label normal : 1 abnormal : 0
data1 = SPECT()

class NaiveBayes:
    def __init__(self, n = 2, prior = 0.5):
        """
        Create a NaiveBayes classifier
        :param n : small integer
        :param prior: prior estimate of the value of pi
        """
        self.n = n
        self.prior = prior
        self.normal_model = None
        self.abnormal_model = None
        self.p_normal = 0
        self.p_abnormal = 0
        
    def fit(self, X_train, y_train):
        """
        Generate probabilistic models for normal and abmornal group.
        Use self.normal_model and self.abnormal_model to store 
        models for normal and abnormal groups respectively
        """
        def probability(num_features_matching_label, total_num_of_label):
            return (num_features_matching_label + self.n * self.prior) / (total_num_of_label + self.n)
            
        self.normal_model = np.zeros(len(X_train[0]))
        self.abnormal_model = np.zeros(len(X_train[0]))
        
        num_normal_label = 0
        for i in range(len(X_train)):
            if y_train[i] == 1:
                num_normal_label += 1
        num_abnormal_label = len(X_train) - num_normal_label
        
        self.p_normal = num_normal_label / len(X_train)
        self.p_abnormal = num_abnormal_label / len(X_train)
        
        for i in range(len(X_train[0])):
            num_normal_feature_for_label = 0
            num_abnormal_feature_for_label = 0
            for j in range(len(X_train)):
                if y_train[j] == 1 and X_train[j,i] == 1:
                    num_normal_feature_for_label += 1
                elif y_train[j] == 0 and X_train[j,i] == 0:
                    num_abnormal_feature_for_label += 1
            self.normal_model[i] = probability(num_normal_feature_for_label, num_normal_label)
            self.abnormal_model[i] = probability(num_abnormal_feature_for_label, num_abnormal_label)
            
        return self
    
    def predict(self, data):
        """
        Return predicted label for the input example
        :param data: input example
        """
        p_normal_given_data = 1
        for i in range(len(self.normal_model)):
            p_normal_given_data *= self.normal_model[i]**data[i] * (1-self.normal_model[i])**(1-data[i])
        p_normal_given_data *= self.p_normal
            
        p_abnormal_given_data = 1
        for i in range(len(self.abnormal_model)):
            p_abnormal_given_data *= self.abnormal_model[i]**(data[i]+1) * (1-self.abnormal_model[i])**(data[i])
        p_abnormal_given_data *= self.p_abnormal
        
        return 1 if p_normal_given_data >= p_abnormal_given_data else 0

# The code below evaluates the error rate from the trained model on the test set

nb = NaiveBayes()
nb = nb.fit(data1.X_train, data1.y_train)

misclassifications = 0
for X,y in zip(data1.X_test, data1.y_test):
    if nb.predict(X) != y:
        misclassifications += 1
print("error rate:\t" + str(misclassifications / len(data1.X_test)))
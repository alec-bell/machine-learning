import numpy as np

# Wrapper to generate 100 random points for 3 different means (-4.0, 3.5, 10.6)
# with 3 respective variances (1.5, 1.2, 1.0). We will use this data for training 
# our Gaussian Mixture Model
class Data1D:
    def __init__(self):
        self.means = [-4.0, 3.5, 10.6]
        self.variances = [1.5, 1.2, 1.0]
        X = []
        for m, v in zip(self.means, self.variances):
            X += list(np.random.normal(m, np.sqrt(v), size=(100)))
        self.means, self.variances = np.array(self.means), np.array(self.variances)
        X = np.array(X)
        self.X = X

class GaussianMixtureModel1D:
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.mean, self.variance, self.weight = self.initialize_parameters()
        
    def initialize_parameters(self):
        mean = np.random.choice(self.X, self.K)
        variance = np.random.random_sample(size=self.K) * 2
        weights = np.ones(self.K) / self.K
        return mean, variance, weights
    
    def compute_pdf(self, x, k):
        '''
        Evaluate the p.d.f value for 1-D point i.e scalar value for the w.r.t to the k-th cluster
        Params : 
            x : (float) the point
            k : (integer) the k-th elements from mean, variance and weights correspond to k-th cluster parameters.
                Use those to estimate your result.
        RETURN :
            result : (float) evalutated using the formula described above
        '''
        from math import sqrt, pi, e
        coeff = 1 / sqrt(2 * pi * self.variance[k])
        exp = -1/2 * (x - self.mean[k])**2 / self.variance[k]
        return coeff * e**exp
       
    
    def compute_pdf_matrix(self):
        '''
        Evaluate the p.d.f martix by calling compute_pdf() for each combination of x and k
        Params : 
            None
        RETURN :
            result : (np.array) matrix of size N X K containing p.d.f values
        '''
        result = np.zeros((len(self.X), self.K))
        for i in range(len(self.X)):
            for j in range(self.K):
                result[i,j] = self.compute_pdf(self.X[i], j)
        return result
                  
    def compute_posterior(self, pdf_matrix):
        '''
        Evaluate the posterior probability matrix as described by the formula above
        Params : 
            pdf_matrix : (np.array) matrix of size N X K containing p.d.f values
        RETURN :
            result : (np.array) matrix of size N X K containing posterior probability values
        '''
        def compute_posterior_row(pdf_row):
            posterior_row = np.zeros(pdf_row.shape)
            denom = 0
            for i in range(len(pdf_row)):
                denom += pdf_row[i]*self.weight[i]
            for i in range(len(pdf_row)):
                posterior_row[i] = pdf_row[i]*self.weight[i] / denom
            return posterior_row
        
        posterior_matrix = np.zeros(pdf_matrix.shape)
        for i in range(len(pdf_matrix)):
            posterior_row = compute_posterior_row(pdf_matrix[i])
            posterior_matrix[i] = posterior_row
        return posterior_matrix
    
    def reestimate_params(self, posterior_matrix):
        '''
        Re-estimate the cluster parameters as described by the formulae above and 
        store them in their respective class variables
        Params : 
            posterior_matrix : (np.array) matrix of size N X K containing posterior probability values
        RETURN :
            None
        '''
        def compute_b_k(K_idx):
            sum = 0
            for i in range(len(posterior_matrix)):
                sum += posterior_matrix[i,K_idx]
            return sum
        
        def mean_and_variance(K_idx, b_k):
            mean_sum = 0
            variance_sum = 0
            for i in range(len(posterior_matrix)):
                mean_sum += posterior_matrix[i,K_idx]*self.X[i]
                variance_sum += posterior_matrix[i,K_idx] * (self.X[i] - self.mean[K_idx])**2
            mean = mean_sum / b_k
            variance = variance_sum / b_k
            return mean, variance
        
        for j in range(self.K):
            b_k = compute_b_k(j)
            self.mean[j], self.variance[j] = mean_and_variance(j, b_k)
            self.weight[j] = b_k * 1/len(self.X)
    
    def exp_maximize(self, epochs):
        '''
        Peform the expectation-maximization method as dicussed above by calling the functions in their 
        respective sequence. Also plot the progress of the process by calling the plot_progress function
        after every regular interval of epochs.
        Params : 
            epochs : (integer) maximum number of epochs to run the loop for
        RETURN :
            None
            '''
        for i in range(epochs):
            pdf_matrix = self.compute_pdf_matrix()
            posterior_matrix = self.compute_posterior(pdf_matrix)
            self.reestimate_params(posterior_matrix)
            self.plot_progress()
    
    def plot_progress(self):
        '''
        Helper method for plotting the progress of the 3 distributions.
        Params :
            None
        RETURN :
            None
        '''
        import matplotlib.pyplot as plt
        points = np.linspace(np.min(self.X),np.max(self.X),500)
        plt.figure(figsize=(10,4))
        plt.xlabel("$x$")
        plt.ylabel("pdf")
        plt.plot(self.X, 0.1*np.ones_like(self.X), 'x', color='navy')
        for k in range(self.K):
            plt.plot(points, [self.compute_pdf(p, k) for p in points])
        plt.show()

# The code below simply trains the GaussianMixtureModel1D above and plots its progress

data1D = Data1D()
gmm = GaussianMixtureModel1D(data1D.X, K=3)
gmm.exp_maximize(20)
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, pairwise_kernels
import matplotlib.pyplot as plt


class SampleGeneration:

    def __init__(self, sample_size=100, dist_type='uniform', parameters=np.array([0, 1]), inverse_cdf=None):
        self.sample_size = sample_size
        self.dist_name = dist_type
        self.parameters = parameters
        self.inverse_cdf = inverse_cdf
        self.samples = np.zeros(self.sample_size)

    def sampling(self):
        # np.random.seed(0)
        if callable(self.inverse_cdf):
            temp_sample = np.random.uniform(self.parameters[0], self.parameters[1], self.sample_size)
            self.samples = self.inverse_cdf(temp_sample)
        elif self.dist_name == 'uniform':
            self.samples = np.random.uniform(self.parameters[0], self.parameters[1], self.sample_size)
        elif self.dist_name == 'normal':
            self.samples = np.random.normal(self.parameters[0], self.parameters[1], self.sample_size)
        else:
            print('no inverse cdf or distribution specified.')
        return self.samples


class KernelInitialization:

    # Initialize the transport map using the source and target samples through Kernel ridge regression.
    # The parameter alpha is the regularization strength.

    def __init__(self, center_samples=0, target_samples=0, kernel=None, kernel_type='polynomial', alpha=0.5, kernel_params=False):
        # Valid values for metric are:
        # [‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’, ‘laplacian’, ‘sigmoid’, ‘cosine’]
        # feature_arr = np.zeros(len(center_samples))
        # self.center_samples = np.stack((center_samples, feature_arr), axis=0)
        self.center_samples = np.array_split(center_samples, len(center_samples))
        self.target_samples = target_samples
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.alpha = alpha
        self.kernel_ridge = None
        self.weights = None
        if not kernel_params:
            self.kernel_params = {'gamma': 1, 'degree': 3, 'coef0': 1}
        else:
            self.kernel_params = kernel_params
        self.gamma = self.kernel_params['gamma']
        self.degree = self.kernel_params['degree']
        self.coef0 = self.kernel_params['coef0']
        self.weights = None

    def set_params(self, params):
        self.kernel_params = params
        self.gamma = self.kernel_params['gamma']
        self.degree = self.kernel_params['degree']
        self.coef0 = self.kernel_params['coef0']
        return self

    def regression_init(self):
        # The outcome function using Kernel ridge regression to initialize the weights.
        if callable(self.kernel):
            print('WIP')
        else:
            kernel_instance = KernelRidge(kernel=self.kernel_type, alpha=self.alpha, gamma=self.gamma, degree=self.degree, coef0=self.coef0, kernel_params=self.kernel_params)
            kernel_instance.fit(self.center_samples, self.target_samples)
            self.weights = kernel_instance .dual_coef_
            # Print the weights (dual coefficients) of the kernel ridge model
            # print("Weights (Dual Coefficients):")
            # print(self.kernel_ridge.dual_coef_)
        return self.weights


class FunctionApprox:

    def __init__(self, center_samples, inputs, weights, kernel=None, kernel_type='polynomial', alpha=0.5, kernel_params=False):

        self.center_samples = np.array_split(center_samples, len(center_samples))
        self.inputs = np.array_split(inputs, len(inputs))
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.alpha = alpha
        self.weights = weights
        if not kernel_params:
            self.kernel_params = {'gamma': 1, 'degree': 3, 'coef0': 1}
        else:
            self.kernel_params = kernel_params
        self.gamma = self.kernel_params['gamma']
        self.degree = self.kernel_params['degree']
        self.coef0 = self.kernel_params['coef0']
        self.weights = weights
        self.k = None

    def get_kernel(self):
        x = self.center_samples
        y = self.inputs
        if callable(self.kernel):
            params = self.kernel_params or {}
            k = pairwise_kernels(y, x, metric=self.kernel, filter_params=True, **params)
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
            k = pairwise_kernels(y, x, metric=self.kernel_type, filter_params=True, **params)
        self.k = k
        return self.k

    def predict(self):
        self.get_kernel()
        return np.dot(self.k, self.weights)


c = SampleGeneration(sample_size=10)
source = c.sampling()
target = c.sampling()
ker = KernelInitialization(source, target)
weights = ker.regression_init()
# print(weights)
inputs = c.sampling()
fun = FunctionApprox(source, inputs, weights)
pre = fun.predict()
# print(pre)
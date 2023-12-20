import numpy as np
from kernalization import SampleGeneration
from kernalization import FunctionApprox
from kernalization import KernelInitialization
from scipy.stats import norm
import math
import scipy.integrate as integrate
from cmath import sqrt
from cmath import exp


class CostFunction:

    def __init__(self, cost_type='quadratic', cost_params=False, cost_function=None):
        self.cost_type = cost_type
        self.cost_function = cost_function
        self.cost_params = cost_params
        # if not cost_params:
        #     self.cost_params = {'coe1': 1}
        # else:
        #     self.cost_params = cost_params
        self.coe1 = 1/2

    def set_params(self, params):
        self.cost_params = params
        # self.coe1 = self.cost_params['coe1']
        return self

    def cost_eval(self, x, y):
        if self.cost_function is None:
            if self.cost_type == 'quadratic':
                # print('yes')
                # print(self.coe1 * (x - y) ** 2)
                return self.coe1 * (x - y) ** 2
            else:
                print('This cost type is not available as a default yet, please provide the exact cost function.')
        else:
            if callable(self.cost_function):
                return self.cost_function(x, y)
            else:
                print('the input cost function is not callable')

    def cost_firstd(self, x, y):
        # first derivative of the cost function with respect to y.
        if self.cost_function is None:
            if self.cost_type == 'quadratic':
                return - 2 * self.coe1 * (x - y)
            else:
                print('This cost type is not available as a default yet, please provide the exact cost function.')
        else:
            print('Sorry, the first derivative is not available now.')

    def cost_secondd(self, x, y):
        # first derivative of the cost function with respect to y.
        if self.cost_function is None:
            if self.cost_type == 'quadratic':
                return 0
            else:
                print('This cost type is not available as a default yet, please provide the exact cost function.')
        else:
            print('Sorry, the second derivative is not available now.')


class ObjectivesEval:

    def __init__(self, cost_type='quadratic', cost_params=False, cost_function=None, kernel_type='polynomial', alpha=0.5, kernel_params=False,  weights=None, center_samples=None, dist_type='uniform', mcsample_size=100, dist_parameters=np.array([0, 1]), inverse_cdf=None):
        # give samples (center of the kernel function) directly or generate it in this class
        self.cost_type = cost_type
        self.cost_params = cost_params
        self.cost_function = cost_function
        self.kernel_type = kernel_type
        self.alpha = alpha
        self.kernel_params = kernel_params
        self.weights = weights
        if center_samples is None:
            center_instance = SampleGeneration(sample_size=len(weights), dist_type=dist_type, parameters=dist_parameters, inverse_cdf=inverse_cdf)
            self.center_samples = center_instance.sampling()
        else:
            self.center_samples = center_samples
        self.dist_type = dist_type
        self.mcsample_size = mcsample_size
        self.dist_parameters = dist_parameters
        self.inverse_cdf = inverse_cdf
        self.time_horizon = np.array([-10000, 10000])

    def objective_fun(self):
        mc_instance = SampleGeneration(sample_size=self.mcsample_size, dist_type=self.dist_type, parameters=self.dist_parameters, inverse_cdf=self.inverse_cdf)
        mc_samples = mc_instance.sampling()
        kernel_instance = FunctionApprox(center_samples=self.center_samples, inputs=mc_samples, weights=self.weights, kernel=None, kernel_type=self.kernel_type, alpha=self.alpha, kernel_params=self.kernel_params)
        predicts = kernel_instance.predict()
        cost_instance = CostFunction(cost_type=self.cost_type, cost_params=self.cost_params, cost_function=self.cost_function)
        cost_array = cost_instance.cost_eval(mc_samples, predicts)
        # print(predicts, predicts.shape)
        # print(cost_array)
        objective = np.average(cost_array)
        return objective

    def first_variation(self):
        mc_instance = SampleGeneration(sample_size=self.mcsample_size, dist_type=self.dist_type,
                                       parameters=self.dist_parameters, inverse_cdf=self.inverse_cdf)
        mc_samples = mc_instance.sampling()
        kernel_instance = FunctionApprox(center_samples=self.center_samples, inputs=mc_samples, weights=self.weights, kernel=None,
                                         kernel_type=self.kernel_type, alpha=self.alpha, kernel_params=self.kernel_params)
        predicts = kernel_instance.predict()
        cost_instance = CostFunction(cost_type=self.cost_type, cost_params=self.cost_params, cost_function=self.cost_function)
        cost_array = cost_instance.cost_firstd(mc_samples, predicts)
        integrand_array = np.matmul(cost_array, kernel_instance.k)
        objective = integrand_array / self.mcsample_size
        return objective

    def second_variation(self):
        mc_instance = SampleGeneration(sample_size=self.mcsample_size, dist_type=self.dist_type,
                                       parameters=self.dist_parameters, inverse_cdf=self.inverse_cdf)
        mc_samples = mc_instance.sampling()
        kernel_instance = FunctionApprox(center_samples=self.center_samples, inputs=mc_samples, weights=self.weights,
                                         kernel=None,
                                         kernel_type=self.kernel_type, alpha=self.alpha,
                                         kernel_params=self.kernel_params)
        predicts = kernel_instance.predict()
        cost_instance = CostFunction(cost_type=self.cost_type, cost_params=self.cost_params,
                                     cost_function=self.cost_function)
        cost_array = cost_instance.cost_secondd(mc_samples, predicts)
        matrix_sum = np.zeros((self.mcsample_size, self.mcsample_size))
        for i in range(self.mcsample_size):
            kernel_vec = np.transpose(kernel_instance.k)
            matrix_two_kernels = np.outer(kernel_vec, kernel_vec)
            matrix_sum += matrix_two_kernels * cost_array[i]
        objective = matrix_sum / self.mcsample_size
        return objective


def normal_first_d(x):
    #standard normal

    return norm.pdf(x) * (-1 * x)


def normal_second_d(x):
    # standard normal

    return norm.pdf(x) * math.exp(-1 * x) * math.exp(-1 * x) - norm.pdf(x)


class ConstraintEval:

    def __init__(self, kernel_type='polynomial', alpha=0.5, kernel_params=False, weights=None, center_samples=None, dist_type='uniform', mcsample_size=10000, dist_parameters=np.array([0, 1]), inverse_cdf=None, x_perturb_size=0.5, w_perturb_size=0.05, target_density=norm.pdf, target_density_first_d=normal_first_d, target_density_second_d=normal_second_d, h_magnitude=1):
        self.kernel_type = kernel_type
        self.alpha = alpha
        self.kernel_params = kernel_params
        self.weights = weights
        if center_samples is None:
            center_instance = SampleGeneration(sample_size=len(self.weights), dist_type=dist_type, parameters=dist_parameters,
                                               inverse_cdf=inverse_cdf)
            self.center_samples = center_instance.sampling()
        else:
            self.center_samples = center_samples
        self.dist_type = dist_type
        self.mcsample_size = mcsample_size
        self.dist_parameters = dist_parameters
        self.inverse_cdf = inverse_cdf
        self.x_perturb_size = x_perturb_size
        self.w_perturb_size = w_perturb_size
        self.target_density = target_density
        self.h_magnitude = h_magnitude
        self.time_horizon = np.array([-10000, 10000])
        self.target_density_first_d = target_density_first_d
        self.target_density_second_d = target_density_second_d

    def fp_operator(self, x):
        mc_instance = SampleGeneration(sample_size=self.mcsample_size, dist_type=self.dist_type,
                                       parameters=self.dist_parameters, inverse_cdf=self.inverse_cdf)
        mc_samples = mc_instance.sampling()
        kernel_instance = FunctionApprox(center_samples=self.center_samples, inputs=mc_samples, weights=self.weights,
                                         kernel=None, kernel_type=self.kernel_type, alpha=self.alpha,
                                         kernel_params=self.kernel_params)
        predicts = kernel_instance.predict()
        # if predicts[0] == np.nan or math.isnan(predicts[0]):
        #     print('hi heres a nan')
        #     print(predicts)
        unperturbed = (predicts < x).sum()
        perturbed = (predicts < x + self.x_perturb_size).sum()
        psf_at_x = (perturbed - unperturbed)/self.x_perturb_size - self.target_density(x)
        return psf_at_x

    def check_feasible(self, weights, domain_grid, cons_tol):
        cons_eval = []
        for i in range(len(domain_grid)):
            x = domain_grid[i]
            mc_instance = SampleGeneration(sample_size=self.mcsample_size, dist_type=self.dist_type,
                                           parameters=self.dist_parameters, inverse_cdf=self.inverse_cdf)
            mc_samples = mc_instance.sampling()
            kernel_instance = FunctionApprox(center_samples=self.center_samples, inputs=mc_samples, weights=weights,
                                             kernel=None, kernel_type=self.kernel_type, alpha=self.alpha,
                                             kernel_params=self.kernel_params)
            predicts = kernel_instance.predict()
            unperturbed = (predicts < x).sum()
            perturbed = (predicts < x + self.x_perturb_size).sum()
            psf_at_x = (perturbed - unperturbed) / self.x_perturb_size - self.target_density(x)
            cons_eval.append(psf_at_x)
        if np.amax(np.absolute(np.array(cons_eval))) < cons_tol:
            print('the initial guess is ', weights)
            print('feasible under tolerance of ', cons_tol)
        else:
            print('the initial guess is ', weights)
            print('not feasible, the constraints values are ', cons_eval)
        return cons_eval

    def fp_fd_first_variation(self, x):
        # using finite differencing on h
        # the objective is the estimated gradient with respect to weights
        objective = np.zeros(len(self.weights))
        mc_instance = SampleGeneration(sample_size=self.mcsample_size, dist_type=self.dist_type,
                                       parameters=self.dist_parameters, inverse_cdf=self.inverse_cdf)
        mc_samples = mc_instance.sampling()
        psf_at_x_w_unperturbed = self.fp_operator(x)
        perturb_weights_init = self.weights
        for i in range(len(self.weights)):
            perturb_weights = perturb_weights_init
            perturb_weights[i] += self.w_perturb_size
            kernel_instance = FunctionApprox(center_samples=self.center_samples, inputs=mc_samples,
                                             weights=perturb_weights,
                                             kernel=None, kernel_type=self.kernel_type, alpha=self.alpha,
                                             kernel_params=self.kernel_params)
            predicts = kernel_instance.predict()
            x_unperturbed = (predicts < x).sum()
            x_perturbed = (predicts < x + self.x_perturb_size).sum()
            psf_at_x_w_perturbed = (x_perturbed - x_unperturbed) / self.x_perturb_size
            objective[i] = (psf_at_x_w_perturbed - psf_at_x_w_unperturbed)/self.w_perturb_size
        return objective

    def fp_fd_second_variation(self, x):

        objective = np.zeros((len(self.weights), len(self.weights)))

        def first_varr(y, weights):
            obj = np.zeros(len(weights))
            mc_instance = SampleGeneration(sample_size=self.mcsample_size, dist_type=self.dist_type,
                                           parameters=self.dist_parameters, inverse_cdf=self.inverse_cdf)
            mc_samples = mc_instance.sampling()
            psf_at_x_w_unperturbed = self.fp_operator(y)
            weights_init = weights
            for i in range(len(weights)):
                perturb_weights = weights_init
                perturb_weights[i] += self.w_perturb_size
                kernel_instance = FunctionApprox(center_samples=self.center_samples, inputs=mc_samples,
                                                 weights=perturb_weights,
                                                 kernel=None, kernel_type=self.kernel_type, alpha=self.alpha,
                                                 kernel_params=self.kernel_params)
                predicts = kernel_instance.predict()
                x_unperturbed = (predicts < x).sum()
                x_perturbed = (predicts < x + self.x_perturb_size).sum()
                psf_at_x_w_perturbed = (x_perturbed - x_unperturbed) / self.x_perturb_size
                obj[i] = (psf_at_x_w_perturbed - psf_at_x_w_unperturbed) / self.w_perturb_size
            return obj
        perturb_weights_init = self.weights
        for j in range(len(self.weights)):
            perturbed_weights = perturb_weights_init
            perturbed_weights[j] += self.w_perturb_size
            psfs_w_perturbed = first_varr(x, perturbed_weights)
            psfs_w_unperturbed = first_varr(x, self.weights)
            # print(j, psfs_w_perturbed, psfs_w_unperturbed)
            objective[j] = (psfs_w_perturbed - psfs_w_unperturbed) / self.w_perturb_size
        return objective

    def fp_quad_first_variation(self, x):
        mc_instance_s = SampleGeneration(sample_size=self.mcsample_size, dist_type=self.dist_type, parameters=self.dist_parameters, inverse_cdf=self.inverse_cdf)
        mc_instance_h = SampleGeneration(sample_size=self.mcsample_size, dist_type=self.dist_type, parameters=self.dist_parameters, inverse_cdf=self.inverse_cdf)
        mc_samples_s = mc_instance_s.sampling()
        mc_samples_h = mc_instance_h.sampling()
        dir_magnitude = math.sqrt(self.h_magnitude ** 2 / len(self.weights))
        kernel_instance_s = FunctionApprox(center_samples=self.center_samples, inputs=mc_samples_s, weights=self.weights, kernel=None, kernel_type=self.kernel_type, alpha=self.alpha, kernel_params=self.kernel_params)
        predicts_s = kernel_instance_s.predict()
        expected_s = np.average(predicts_s)
        expected_h = np.zeros(len(self.weights))
        objective = np.zeros(len(self.weights))
        integration_error = np.zeros(len(self.weights))
        for i in range(len(self.weights)):
            direction = np.zeros(len(self.weights))
            # print(np.real(dir_magnitude))
            direction[i] = np.real(dir_magnitude)
            kernel_instance_h = FunctionApprox(center_samples=self.center_samples, inputs=mc_samples_h, weights=direction, kernel=None, kernel_type=self.kernel_type, alpha=self.alpha, kernel_params=self.kernel_params)
            predicts_h_entry = kernel_instance_h.predict()
            expected_h_entry = np.average(predicts_h_entry)
            expected_h[i] = expected_h_entry

            def expected_long_entry(t):
                return np.average(1j * t * np.exp(1j * t * (predicts_s - expected_s)) * (predicts_h_entry - expected_h_entry))
            # expected_long.append(expected_long_entry)
            # print(predicts_h_entry, expected_h)

            def integrand_eval(t):
                integrand = - 1 / (2 * math.pi) * np.exp(1j * t * x) * np.exp(- expected_s) * expected_long_entry(t) - self.target_density_first_d(x) * expected_h_entry
                # integrand = -np.exp(1j * t * x) * np.exp(- expected_s) * t - self.target_density(x) * expected_h_entry
                return integrand
            obj_entry = integrate.quad(integrand_eval, self.time_horizon[0], self.time_horizon[1])
            integration_error[i] = obj_entry[1]
            objective[i] = obj_entry[0]
        return objective

    def fp_quad_second_variation(self, x):
        mc_instance_s = SampleGeneration(sample_size=self.mcsample_size, dist_type=self.dist_type, parameters=self.dist_parameters, inverse_cdf=self.inverse_cdf)
        mc_instance_h = SampleGeneration(sample_size=self.mcsample_size, dist_type=self.dist_type, parameters=self.dist_parameters, inverse_cdf=self.inverse_cdf)
        mc_samples_s = mc_instance_s.sampling()
        mc_samples_h = mc_instance_h.sampling()
        dir_magnitude = math.sqrt(self.h_magnitude ** 2 / len(self.weights))
        kernel_instance_s = FunctionApprox(center_samples=self.center_samples, inputs=mc_samples_s, weights=self.weights, kernel=None, kernel_type=self.kernel_type, alpha=self.alpha, kernel_params=self.kernel_params)
        predicts_s = kernel_instance_s.predict()
        expected_s = np.average(predicts_s)
        expected_h = np.zeros(len(self.weights))
        objective = np.zeros(len(self.weights))
        integration_error = np.zeros(len(self.weights))
        for i in range(len(self.weights)):
            direction = np.zeros(len(self.weights))
            # print(np.real(dir_magnitude))
            direction[i] = np.real(dir_magnitude)
            kernel_instance_h = FunctionApprox(center_samples=self.center_samples, inputs=mc_samples_h, weights=direction, kernel=None, kernel_type=self.kernel_type, alpha=self.alpha, kernel_params=self.kernel_params)
            predicts_h_entry = kernel_instance_h.predict()
            expected_h_entry = np.average(predicts_h_entry)
            expected_h[i] = expected_h_entry

            def expected_long_entry(t):
                return np.average(np.exp(1j * t * (predicts_s - expected_s)) * (predicts_h_entry - expected_h_entry) * (t ** 2 * (predicts_h_entry - expected_h_entry) + 1j * expected_h_entry))
            # expected_long.append(expected_long_entry)
            # print(predicts_h_entry, expected_h)

            def integrand_eval(t):
                integrand = 1 / (2 * math.pi) * np.exp(1j * t * x) * np.exp(- expected_s) * expected_long_entry(t) + self.target_density_second_d(x) * expected_h_entry ** 2
                # integrand = -np.exp(1j * t * x) * np.exp(- expected_s) * t - self.target_density(x) * expected_h_entry
                return integrand
            obj_entry = integrate.quad(integrand_eval, self.time_horizon[0], self.time_horizon[1])
            integration_error[i] = obj_entry[1]
            objective[i] = obj_entry[0]
        return objective

    def fp_first_variation(self, t_arr=True):
        """

        :param t_arr: the t_array which is the frequency array
        :return: a k x len(t_arr) numpy array where k is the length of the gradient vector, number of kernels.
        len(t_arr) is the length of the inverse discrete fourier transform.
        """

        if t_arr:
            arr1 = np.arange(0, 11, 1)
            arr2 = np.arange(-10, 0, 1)
            t_arr = np.concatenate((arr1, arr2))
            delta_t = 1
            init_t = arr2[0]
        mc_instance_s = SampleGeneration(sample_size=self.mcsample_size, dist_type=self.dist_type, parameters=self.dist_parameters, inverse_cdf=self.inverse_cdf)
        mc_instance_h = SampleGeneration(sample_size=self.mcsample_size, dist_type=self.dist_type, parameters=self.dist_parameters, inverse_cdf=self.inverse_cdf)
        mc_samples_s = mc_instance_s.sampling()
        mc_samples_h = mc_instance_h.sampling()
        dir_magnitude = math.sqrt(self.h_magnitude ** 2 / len(self.weights))
        kernel_instance_s = FunctionApprox(center_samples=self.center_samples, inputs=mc_samples_s, weights=self.weights, kernel=None, kernel_type=self.kernel_type, alpha=self.alpha, kernel_params=self.kernel_params)
        predicts_s = kernel_instance_s.predict()
        expected_s = np.average(predicts_s)
        expected_h = np.zeros(len(self.weights))
        objective = []
        integration_error = np.zeros(len(self.weights))
        for i in range(len(self.weights)):
            # for each i, objective[i] would be a len(t_arr) long np array.

            direction = np.zeros(len(self.weights))
            # print(np.real(dir_magnitude))
            direction[i] = np.real(dir_magnitude)
            kernel_instance_h = FunctionApprox(center_samples=self.center_samples, inputs=mc_samples_h, weights=direction, kernel=None, kernel_type=self.kernel_type, alpha=self.alpha, kernel_params=self.kernel_params)
            predicts_h_entry = kernel_instance_h.predict()
            expected_h_entry = np.average(predicts_h_entry)
            expected_h[i] = expected_h_entry

            def expected_long_entry(t):
                return np.average(1j * t * np.exp(1j * t * (predicts_s - expected_s)) * (predicts_h_entry - expected_h_entry))
            # expected_long.append(expected_long_entry)
            # print(predicts_h_entry, expected_h)

            def integrand_eval(t):
                integrand = - np.exp(- expected_s) * expected_long_entry(t)
                if math.isnan(np.real(integrand)) or math.isinf(np.real(integrand)):
                    print("integrand has issue", integrand, "t is", t)
                    print("expected_s is", expected_s)
                    print("expected_long_entry(t) is", expected_long_entry(t))
                return integrand

            a_arr = np.zeros(len(t_arr), dtype='complex64')
            for j in range(len(t_arr)):
                a_arr[j] = integrand_eval(t_arr[j])
            # if np.isinf(a_arr).sum() > 0:
            #     print("a_arr is", a_arr)
            capital_a_arr = np.fft.ifft(a_arr)
            ifft_arr = capital_a_arr * delta_t * exp(1j * 2 * math.pi * init_t / (len(t_arr) * delta_t))

            g_prime_term = np.zeros(len(t_arr))
            # not sure about this
            for j in range(len(t_arr)):
                x = (j - len(t_arr)/2) / (len(t_arr)/2 * delta_t)
                g_prime_term[j] = self.target_density_first_d(x) * expected_h_entry
            objective.append(ifft_arr + g_prime_term)
        x_arr = np.arange(0, len(t_arr), 1)
        return [np.array(objective), x_arr]

    def fp_second_variation(self, t_arr=True):
        """

        :param t_arr: the t_array which is the frequency array
        :return: a k x k x len(t_arr) numpy array where k is the length of the gradient vector, number of kernels.
        len(t_arr) is the length of the inverse discrete fourier transform.
        """

        if t_arr:
            arr1 = np.arange(0, 11, 1)
            arr2 = np.arange(-10, 0, 1)
            t_arr = np.concatenate((arr1, arr2))
            delta_t = 1
            init_t = arr2[0]

        mc_instance_s = SampleGeneration(sample_size=self.mcsample_size, dist_type=self.dist_type, parameters=self.dist_parameters, inverse_cdf=self.inverse_cdf)
        mc_instance_h = SampleGeneration(sample_size=self.mcsample_size, dist_type=self.dist_type, parameters=self.dist_parameters, inverse_cdf=self.inverse_cdf)
        mc_samples_s = mc_instance_s.sampling()
        mc_samples_h = mc_instance_h.sampling()
        dir_magnitude = math.sqrt(self.h_magnitude ** 2 / len(self.weights))
        kernel_instance_s = FunctionApprox(center_samples=self.center_samples, inputs=mc_samples_s, weights=self.weights, kernel=None, kernel_type=self.kernel_type, alpha=self.alpha, kernel_params=self.kernel_params)
        predicts_s = kernel_instance_s.predict()
        expected_s = np.average(predicts_s)
        expected_h1 = np.zeros(len(self.weights))
        expected_h2 = np.zeros(len(self.weights))
        objective = np.zeros((len(self.weights), len(self.weights), len(t_arr)), dtype='complex64')
        for i in range(len(self.weights)):
            direction1 = np.zeros(len(self.weights))
            # print(np.real(dir_magnitude))
            direction1[i] = np.real(dir_magnitude)
            kernel_instance_h1 = FunctionApprox(center_samples=self.center_samples, inputs=mc_samples_h, weights=direction1, kernel=None, kernel_type=self.kernel_type, alpha=self.alpha, kernel_params=self.kernel_params)
            predicts_h1_entry = kernel_instance_h1.predict()
            expected_h1_entry = np.average(predicts_h1_entry)
            expected_h1[i] = expected_h1_entry
            for j in range(len(self.weights)):
                # computing the derivative wrt w_i and w_j for an array of x
                direction2 = np.zeros(len(self.weights))
                # print(np.real(dir_magnitude))
                direction2[i] = np.real(dir_magnitude)
                kernel_instance_h2 = FunctionApprox(center_samples=self.center_samples, inputs=mc_samples_h,
                                                    weights=direction2, kernel=None, kernel_type=self.kernel_type,
                                                    alpha=self.alpha, kernel_params=self.kernel_params)
                predicts_h2_entry = kernel_instance_h2.predict()
                expected_h2_entry = np.average(predicts_h2_entry)
                expected_h2[i] = expected_h2_entry

                def expected_long_entry(t):
                    return np.average(np.exp(1j * t * (predicts_s - expected_s)) * (predicts_h1_entry - expected_h1_entry) * (t ** 2 * (predicts_h2_entry - expected_h2_entry) + 1j * expected_h2_entry))
            # expected_long.append(expected_long_entry)
            # print(predicts_h_entry, expected_h)

                def integrand_eval(t):
                    integrand = np.exp(- expected_s) * expected_long_entry(t)
                    # integrand = -np.exp(1j * t * x) * np.exp(- expected_s) * t - self.target_density(x) * expected_h_entry
                    return integrand

                a_arr = np.zeros(len(t_arr), dtype='complex64')
                for ele in range(len(t_arr)):
                    a_arr[ele] = integrand_eval(t_arr[ele])
                capital_a_arr = np.fft.ifft(a_arr)
                ifft_arr = capital_a_arr * delta_t * exp(1j * 2 * math.pi * init_t / (len(t_arr) * delta_t))

                g_prime_term = np.zeros(len(t_arr))
                # not sure about this
                for ele in range(len(t_arr)):
                    x = (j - len(t_arr) / 2) / (len(t_arr) / 2 * delta_t)
                    g_prime_term[ele] = self.target_density_second_d(x) * expected_h1_entry * expected_h2_entry
                objective[i][j] = ifft_arr + g_prime_term

        x_arr = np.arange(0, len(t_arr), 1)
        return [objective, x_arr]


# c = SampleGeneration(sample_size=10)
# source = c.sampling()
# target = c.sampling()
# ker = KernelInitialization(source, target)
# weight = ker.regression_init()
#
# source = np.array([1])
# weight = np.array([1])
# c = SampleGeneration(sample_size=100)
# inputs = c.sampling()
# fun = FunctionApprox(source, inputs, weight)
# pre = fun.predict()
# print(weight.shape, source.shape, inputs.shape, pre.shape)
# obj = ObjectivesEval(weights=weight)
# fun_val = obj.objective_fun()
# first = obj.first_variation()
# cons = ConstraintEval(weights=weight)
# psf = cons.fp_first_variation(1)
# print(weight)
# psf = cons.fp_operator()
# psfs = cons.fp_first_variation()
# psfss = cons.fp_second_variation()
# print(psfs, psfss)
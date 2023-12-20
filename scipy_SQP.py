import numpy as np
import math
import random
from kernalization import SampleGeneration
from kernalization import KernelInitialization
from evaluation import ObjectivesEval
from evaluation import ConstraintEval
from scipy.optimize import minimize, NonlinearConstraint


def constraint_grad(weights):
    cons = ConstraintEval(weights=weights)
    grad_cons_at_grid = np.real(cons.fp_first_variation()[0])
    return np.transpose(grad_cons_at_grid)


def constraint_hessian(weights):
    cons = ConstraintEval(weights=weights)
    hessian_cons_at_grid = np.real(cons.fp_second_variation()[0])
    return hessian_cons_at_grid


class ScipySLSQP:

    def __init__(self, kernel_dim=10, source_dist_type='normal', source_parameters=np.array([0, 1]), source_inverse_cdf=None, target_dist_type='normal', target_parameters=np.array([0, 1]), target_inverse_cdf=None, kernel_type='polynomial', alpha=0.5, kernel_params=False, cost_type='quadratic', mcsample_size=100, threshold=10, sqp_step=0.01, domain_grid=np.arange(-5, 6, 1), lagrangian_init=np.ones(101)):
        self.kernel_dim = kernel_dim
        self.source_dist_type = source_dist_type
        self.source_parameters = source_parameters
        self.source_inverse_cdf = source_inverse_cdf
        self.target_dist_type = target_dist_type
        self.target_parameters = target_parameters
        self.target_inverse_cdf = target_inverse_cdf
        self.kernel_type = kernel_type
        self.alpha = alpha
        self.kernel_params = kernel_params
        self.cost_type = cost_type
        self.mcsample_size = mcsample_size
        self.threshold = threshold
        self.sqp_step = sqp_step
        self.domain_grid = domain_grid
        self.lagrangian_init = lagrangian_init
        self.weight_init = None
        self.center_sample = None
        self.initialized = False

    def initialization(self, cons_tol):
        # source_sample_instance = SampleGeneration(sample_size=self.kernel_dim, dist_type=self.source_dist_type,
        #                                           parameters=self.source_parameters,
        #                                           inverse_cdf=self.source_inverse_cdf)
        target_sample_instance = SampleGeneration(sample_size=self.kernel_dim, dist_type=self.target_dist_type,
                                                  parameters=self.target_parameters,
                                                  inverse_cdf=self.target_inverse_cdf)

        source_sample = np.arange(- (self.kernel_dim // 2), self.kernel_dim//2 + 1, 1)
        # print(self.kernel_dim // 2, source_sample)
        target_sample = target_sample_instance.sampling()
        weights_init_instance = KernelInitialization(center_samples=source_sample, target_samples=target_sample,
                                                     kernel=None, kernel_type=self.kernel_type, alpha=self.alpha,
                                                     kernel_params=self.kernel_params)
        weights_init = weights_init_instance.regression_init()
        self.weight_init = weights_init
        cons_instance = ConstraintEval(weights=weights_init)
        cons_instance.check_feasible(weights_init, cons_tol)
        self.center_sample = source_sample
        return self

    def objective_fun_val(self, weights):
        # print(weights)
        obj_instance = ObjectivesEval(cost_type=self.cost_type, kernel_type=self.kernel_type, alpha=self.alpha,
                                      kernel_params=self.kernel_params, weights=weights,
                                      center_samples=self.center_sample, dist_type=self.source_dist_type,
                                      mcsample_size=self.mcsample_size, dist_parameters=self.source_parameters,
                                      inverse_cdf=self.source_inverse_cdf)
        obj_val = obj_instance.objective_fun()
        return obj_val

    def gradient(self, weights):
        obj_instance = ObjectivesEval(cost_type=self.cost_type, kernel_type=self.kernel_type, alpha=self.alpha,
                                      kernel_params=self.kernel_params, weights=weights,
                                      center_samples=self.center_sample, dist_type=self.source_dist_type,
                                      mcsample_size=self.mcsample_size, dist_parameters=self.source_parameters,
                                      inverse_cdf=self.source_inverse_cdf)
        grad_obj = obj_instance.first_variation()
        return grad_obj

    def hessian(self, weights):
        # obj_instance = ObjectivesEval(cost_type=self.cost_type, kernel_type=self.kernel_type, alpha=self.alpha,
        #                               kernel_params=self.kernel_params, weights=weights,
        #                               center_samples=self.center_sample, dist_type=self.source_dist_type,
        #                               mcsample_size=self.mcsample_size, dist_parameters=self.source_parameters,
        #                               inverse_cdf=self.source_inverse_cdf)
        obj_hessian = np.zeros((self.kernel_dim, self.kernel_dim))
        return obj_hessian

    def constraint(self, weights):
        cons = ConstraintEval(weights=weights)
        constraint = np.zeros(len(self.domain_grid))
        for i in range(len(self.domain_grid)):
            constraint[i] = cons.fp_operator(self.domain_grid[i])
        return constraint

    # def dict_gen(self, weights):
    #     cons = self.constraint(weights)
    #     cons_grad = constraint_grad(weights)
    #     li = []
    #     for i in range(cons_grad.shape[0]):
    #         li.append({'type': 'eq', 'fun': lambda x: cons[i], 'jac': lambda x: constraint_grad[i]})
    #     return li

    def sqp_implement(self):
        # self.initialization(1.0)
        self.weight_init = np.zeros(self.kernel_dim)
        init_cons = self.constraint(self.weight_init)
        # init_cons_grad = constraint_grad(self.weight_init)
        # print(init_cons.shape, init_cons_grad.shape)
        li = []
        # print(len(init_cons))
        for i in range(len(init_cons)):
            li.append({'type': 'eq', 'fun': lambda x: self.constraint(x)[i], 'jac': lambda x: constraint_grad(x)[i]})
            # li.append({'type': 'eq', 'fun': lambda x: self.constraint(x)[i]})
        # print(li)
        # cons = ({'type': 'eq', 'fun': self.constraint, 'jac': constraint_grad})
        init_val = self.objective_fun_val(self.weight_init)
        init_grad = self.gradient(self.weight_init)
        # print(self.weight_init)
        # print(init_val, init_grad)
        # res = minimize(self.objective_fun_val, self.weight_init, method='SLSQP', jac=self.gradient, hess=self.hessian,
        #                constraints=cons)
        # res = minimize(self.objective_fun_val, self.weight_init, method='SLSQP', jac=self.gradient, hess=self.hessian)
        res = minimize(self.objective_fun_val, self.weight_init, method='SLSQP', jac=self.gradient, constraints=li)
        # res = minimize(self.objective_fun_val, self.weight_init, method='SLSQP', constraints=li)
        print(res)
        return res


random.seed(10)
instance = ScipySLSQP(kernel_dim=13)
instance.sqp_implement()
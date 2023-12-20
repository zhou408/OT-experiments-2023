import numpy as np
import random
from scipy.stats import norm
from kernalization import SampleGeneration
from kernalization import KernelInitialization
from evaluation import ObjectivesEval
from evaluation import ConstraintEval


def sqp(kernel_dim, source_dist_type='normal', source_parameters=np.array([1, 1]), source_inverse_cdf=None, target_density=norm.pdf, target_density_first_d=None, target_density_second_d=None, kernel_type='linear', alpha=0.5, kernel_params=False, cost_type='quadratic', mcsample_size=300, threshold=0.01, min_ite=10, sqp_step=1e-2, kernel_center=(-5, 6, 0.1), domain_grid=np.arange(-2, 3, 1), lagrangian_init=True):

    # source_sample_instance = SampleGeneration(sample_size=kernel_dim, dist_type=source_dist_type,
    #                                           parameters=source_parameters, inverse_cdf=source_inverse_cdf)
    # target_sample_instance = SampleGeneration(sample_size=kernel_dim, dist_type=target_dist_type,
    #                                           parameters=target_parameters, inverse_cdf=target_inverse_cdf)
    # source_sample = source_sample_instance.sampling()
    # target_sample = target_sample_instance.sampling()
    # weights_init_instance = KernelInitialization(center_samples=source_sample, target_samples=target_sample,
    #                                              kernel=None, kernel_type=kernel_type, alpha=alpha,
    #                                              kernel_params=kernel_params)
    # weights_init = weights_init_instance.regression_init()
    if lagrangian_init:
        lagrangian_init = np.ones(len(domain_grid))
    source_sample = kernel_center
    weights_init = np.zeros(kernel_dim)
    weights_now = weights_init
    lagrangian_now = lagrangian_init
    obj_init_instance = ObjectivesEval(cost_type=cost_type, kernel_type=kernel_type, alpha=alpha,
                                       kernel_params=kernel_params, weights=weights_init, center_samples=source_sample,
                                       dist_type=source_dist_type, mcsample_size=mcsample_size,
                                       dist_parameters=source_parameters, inverse_cdf=source_inverse_cdf)
    obj_init = obj_init_instance.objective_fun()
    lagrangian_list = [lagrangian_init]
    obj_list = [obj_init]
    weight_list = [weights_init, obj_init]
    ite = 0
    print('ite:', ite, ' obj:', obj_init)
    while (ite < min_ite) or (obj_list[ite] - obj_list[ite - 1]) > threshold:
        cons = ConstraintEval(weights=weights_now, domain_grid=domain_grid, target_density=target_density)
        constraint = np.zeros(len(domain_grid))
        grad_constraint = np.zeros((len(domain_grid), kernel_dim))
        obj_instance = ObjectivesEval(cost_type=cost_type, kernel_type=kernel_type, alpha=alpha,
                                      kernel_params=kernel_params, weights=weights_now, center_samples=source_sample,
                                      dist_type=source_dist_type, mcsample_size=mcsample_size,
                                      dist_parameters=source_parameters, inverse_cdf=source_inverse_cdf)
        grad_obj = obj_instance.first_variation()
        # second variation is 0 for quadratic cost function
        # hessian_obj = obj_instance.second_variation()
        hessian_obj = np.zeros((kernel_dim, kernel_dim))
        for i in range(len(domain_grid)):
            constraint[i] = cons.fp_operator(domain_grid[i])
        grad_cons_at_grid = np.real(cons.fp_first_variation()[0])
        # print('grad is ', grad_cons_at_grid)
        grad_cons = np.zeros(kernel_dim)
        for i in range(kernel_dim):
            # print(lagrangian_now, grad_cons_at_grid[i])
            grad_cons[i] = np.matmul(grad_cons_at_grid[i], lagrangian_now)
        grad_lagrangian = grad_obj + grad_cons
        psi = np.concatenate((grad_lagrangian, constraint), axis=0)
        hessian_cons = np.zeros((kernel_dim, kernel_dim))
        hessian_cons_at_grid = np.real(cons.fp_second_variation()[0])
        # print('hessian is', hessian_cons_at_grid)
        for i in range(kernel_dim):
            for j in range(kernel_dim):
                # print(lagrangian_now, hessian_cons_at_grid[i][j])
                hessian_cons[i][j] = np.matmul(lagrangian_now, hessian_cons_at_grid[i][j])
        hessian = hessian_obj + hessian_cons
        # print(hessian.shape, grad_cons_at_grid.shape)
        arr1 = np.column_stack((hessian, grad_cons_at_grid))
        arr2 = np.column_stack((np.transpose(grad_cons_at_grid), np.zeros((len(domain_grid), len(domain_grid)))))
        jacobian = np.row_stack((arr1, arr2))
        # Avoiding singular matrix
        jacobian[jacobian == 0] = 1e-100
        # print(jacobian, -psi)
        sol = np.linalg.solve(jacobian, -psi)
        dw = sol[:kernel_dim]
        dphi = sol[kernel_dim:]
        step_w = 1 / np.linalg.norm(dw) * sqp_step
        step_phi = 1 / np.linalg.norm(dphi) * sqp_step
        weights_now -= step_w * dw
        lagrangian_now -= step_phi * dphi
        obj_instance = ObjectivesEval(cost_type=cost_type, kernel_type=kernel_type, alpha=alpha,
                                      kernel_params=kernel_params, weights=weights_now, center_samples=source_sample,
                                      dist_type=source_dist_type, mcsample_size=mcsample_size,
                                      dist_parameters=source_parameters, inverse_cdf=source_inverse_cdf)
        obj = obj_instance.objective_fun()
        weight_list.append(weights_now)
        lagrangian_list.append(lagrangian_now)
        obj_list.append(obj)
        ite += 1
        print('ite:', ite, ' obj:', obj, 'weights:', weights_now)
        print('direction:', dw)
    return [np.array(obj_list), np.array(weight_list), np.array(lagrangian_list)]


random.seed(10)
sqp(kernel_dim=3)
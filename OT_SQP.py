import numpy as np
from kernalization import SampleGeneration
from kernalization import KernelInitialization
from evaluation import ObjectivesEval
from evaluation import ConstraintEval


def sqp_fd(kernel_dim, source_dist_type='uniform', source_parameters=np.array([0, 1]), source_inverse_cdf=None, target_dist_type='uniform', target_parameters=np.array([0, 1]), target_inverse_cdf=None, kernel_type='polynomial', alpha=0.5, kernel_params=False, cost_type='quadratic', mcsample_size=100, threshold=10, sqp_step=0.01, domain_grid=np.arange(-998, 1002, 2), lagrangian_init=np.ones(1000)):

    source_sample_instance = SampleGeneration(sample_size=kernel_dim, dist_type=source_dist_type,
                                              parameters=source_parameters, inverse_cdf=source_inverse_cdf)
    target_sample_instance = SampleGeneration(sample_size=kernel_dim, dist_type=target_dist_type,
                                              parameters=target_parameters, inverse_cdf=target_inverse_cdf)
    source_sample = source_sample_instance.sampling()
    target_sample = target_sample_instance.sampling()
    weights_init_instance = KernelInitialization(center_samples=source_sample, target_samples=target_sample,
                                                 kernel=None, kernel_type=kernel_type, alpha=alpha,
                                                 kernel_params=kernel_params)
    weights_init = weights_init_instance.regression_init()
    weights_now = weights_init
    lagrangian_now = lagrangian_init
    obj_init_instance = ObjectivesEval(cost_type=cost_type, kernel_type=kernel_type, alpha=alpha,
                                       kernel_params=kernel_params, weights=weights_init, center_samples=source_sample,
                                       dist_type=source_dist_type, mcsample_size=mcsample_size,
                                       dist_parameters=source_parameters, inverse_cdf=source_inverse_cdf)
    obj_instance = obj_init_instance
    obj_init = obj_init_instance.objective_fun()
    lagrangian_list = [lagrangian_init]
    obj_list = [obj_init]
    weight_list = [weights_init]
    ite = 0
    while (ite < 2) or (obj_list[ite] - obj_list[ite - 1]) > threshold:
        cons = ConstraintEval(weights=weights_now)
        constraint = np.zeros(len(domain_grid))
        grad_constraint = np.zeros((len(domain_grid), kernel_dim))
        obj_instance = ObjectivesEval(cost_type=cost_type, kernel_type=kernel_type, alpha=alpha,
                                      kernel_params=kernel_params, weights=weights_now, center_samples=source_sample,
                                      dist_type=source_dist_type, mcsample_size=mcsample_size,
                                      dist_parameters=source_parameters, inverse_cdf=source_inverse_cdf)
        grad_obj = obj_instance.first_variation()
        for i in range(len(domain_grid)):
            constraint[i] = cons.fp_operator(domain_grid[i])
            grad_constraint[i] = cons.fp_fd_first_variation(domain_grid[i])
        grad_lagrangian = grad_obj + np.matmul(np.transpose(grad_constraint), lagrangian_now)
        psi = np.concatenate((grad_lagrangian, constraint), axis=0)
        hessian = np.zeros((kernel_dim, kernel_dim))
        for k in range(len(domain_grid)):
            cons_hessian_at_grid = cons.fp_fd_second_variation(domain_grid[k])
            hessian_at_grid = cons_hessian_at_grid * lagrangian_now[k]
            hessian += hessian_at_grid
        arr1 = np.concatenate(hessian, np.transpose(grad_constraint), axis=1)
        arr2 = np.concatenate(grad_constraint, np.zeros((len(domain_grid), len(domain_grid))), axis=1)
        jacobian = np.concatenate(arr1, arr2, axis=0)
        sol = np.linalg.solve(jacobian, -psi)
        dw = sol[:kernel_dim+1]
        dphi = sol[kernel_dim+1:]
        weights_now -= sqp_step * dw
        lagrangian_now -= sqp_step * dphi
        obj_instance = ObjectivesEval(cost_type=cost_type, kernel_type=kernel_type, alpha=alpha,
                                      kernel_params=kernel_params, weights=weights_now, center_samples=source_sample,
                                      dist_type=source_dist_type, mcsample_size=mcsample_size,
                                      dist_parameters=source_parameters, inverse_cdf=source_inverse_cdf)
        obj = obj_instance.objective_fun()
        weight_list.append(weights_now)
        lagrangian_list.append(lagrangian_now)
        obj_list.append(obj)
        ite += 1
        print('ite:', ite, ' obj:', obj)
        print('direction:', dw)
    return [np.array(obj_list), np.array(weight_list), np.array(lagrangian_list)]


def sqp(kernel_dim, source_dist_type='uniform', source_parameters=np.array([0, 1]), source_inverse_cdf=None, target_dist_type='uniform', target_parameters=np.array([0, 1]), target_inverse_cdf=None, kernel_type='polynomial', alpha=0.5, kernel_params=False, cost_type='quadratic', mcsample_size=100, threshold=10, sqp_step=0.01, domain_grid=np.arange(-998, 1002, 2), lagrangian_init=np.ones(1000)):

    source_sample_instance = SampleGeneration(sample_size=kernel_dim, dist_type=source_dist_type,
                                              parameters=source_parameters, inverse_cdf=source_inverse_cdf)
    target_sample_instance = SampleGeneration(sample_size=kernel_dim, dist_type=target_dist_type,
                                              parameters=target_parameters, inverse_cdf=target_inverse_cdf)
    source_sample = source_sample_instance.sampling()
    target_sample = target_sample_instance.sampling()
    weights_init_instance = KernelInitialization(center_samples=source_sample, target_samples=target_sample,
                                                 kernel=None, kernel_type=kernel_type, alpha=alpha,
                                                 kernel_params=kernel_params)
    weights_init = weights_init_instance.regression_init()
    weights_now = weights_init
    lagrangian_now = lagrangian_init
    obj_init_instance = ObjectivesEval(cost_type=cost_type, kernel_type=kernel_type, alpha=alpha,
                                       kernel_params=kernel_params, weights=weights_init, center_samples=source_sample,
                                       dist_type=source_dist_type, mcsample_size=mcsample_size,
                                       dist_parameters=source_parameters, inverse_cdf=source_inverse_cdf)
    obj_instance = obj_init_instance
    obj_init = obj_init_instance.objective_fun()
    lagrangian_list = [lagrangian_init]
    obj_list = [obj_init]
    weight_list = [weights_init]
    ite = 0
    while (ite < 2) or (obj_list[ite] - obj_list[ite - 1]) > threshold:
        cons = ConstraintEval(weights=weights_now)
        constraint = np.zeros(len(domain_grid))
        grad_constraint = np.zeros((len(domain_grid), kernel_dim))
        obj_instance = ObjectivesEval(cost_type=cost_type, kernel_type=kernel_type, alpha=alpha,
                                      kernel_params=kernel_params, weights=weights_now, center_samples=source_sample,
                                      dist_type=source_dist_type, mcsample_size=mcsample_size,
                                      dist_parameters=source_parameters, inverse_cdf=source_inverse_cdf)
        grad_obj = obj_instance.first_variation()
        for i in range(len(domain_grid)):
            constraint[i] = cons.fp_operator(domain_grid[i])
            grad_constraint[i] = cons.fp_first_variation(domain_grid[i])
        grad_lagrangian = grad_obj + np.matmul(np.transpose(grad_constraint), lagrangian_now)
        psi = np.concatenate((grad_lagrangian, constraint), axis=0)
        hessian = np.zeros((kernel_dim, kernel_dim))
        for k in range(len(domain_grid)):
            cons_hessian_at_grid = cons.fp_second_variation(domain_grid[k])
            hessian_at_grid = cons_hessian_at_grid * lagrangian_now[k]
            hessian += hessian_at_grid
        arr1 = np.concatenate(hessian, np.transpose(grad_constraint), axis=1)
        arr2 = np.concatenate(grad_constraint, np.zeros((len(domain_grid), len(domain_grid))), axis=1)
        jacobian = np.concatenate(arr1, arr2, axis=0)
        sol = np.linalg.solve(jacobian, -psi)
        dw = sol[:kernel_dim+1]
        dphi = sol[kernel_dim+1:]
        weights_now -= sqp_step * dw
        lagrangian_now -= sqp_step * dphi
        obj_instance = ObjectivesEval(cost_type=cost_type, kernel_type=kernel_type, alpha=alpha,
                                      kernel_params=kernel_params, weights=weights_now, center_samples=source_sample,
                                      dist_type=source_dist_type, mcsample_size=mcsample_size,
                                      dist_parameters=source_parameters, inverse_cdf=source_inverse_cdf)
        obj = obj_instance.objective_fun()
        weight_list.append(weights_now)
        lagrangian_list.append(lagrangian_now)
        obj_list.append(obj)
        ite += 1
        print('ite:', ite, ' obj:', obj)
        print('direction:', dw)
    return [np.array(obj_list), np.array(weight_list), np.array(lagrangian_list)]


sqp_fd(kernel_dim=10)
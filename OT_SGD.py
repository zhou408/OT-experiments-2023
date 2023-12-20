import numpy as np
from kernalization import SampleGeneration
from kernalization import KernelInitialization
from evaluation import ObjectivesEval


def gd(kernel_dim, source_dist_type='uniform', source_parameters=np.array([0, 1]), source_inverse_cdf=None, target_dist_type='uniform', target_parameters=np.array([0, 1]), target_inverse_cdf=None, kernel_type='polynomial', alpha=0.5, kernel_params=False, cost_type='quadratic', mcsample_size=100, threshold=10, gd_step=0.001):

    source_sample_instance = SampleGeneration(sample_size=kernel_dim, dist_type=source_dist_type, parameters=source_parameters, inverse_cdf=source_inverse_cdf)
    target_sample_instance = SampleGeneration(sample_size=kernel_dim, dist_type=target_dist_type,
                                              parameters=target_parameters, inverse_cdf=target_inverse_cdf)
    source_sample = source_sample_instance.sampling()
    target_sample = target_sample_instance.sampling()
    weights_init_instance = KernelInitialization(center_samples=source_sample, target_samples=target_sample, kernel=None, kernel_type=kernel_type, alpha=alpha, kernel_params=kernel_params)
    weights_init = weights_init_instance.regression_init()
    weights_now = weights_init
    obj_init_instance = ObjectivesEval(cost_type=cost_type, kernel_type=kernel_type, alpha=alpha, kernel_params=kernel_params,  weights=weights_init, center_samples=source_sample, dist_type=source_dist_type, mcsample_size=mcsample_size, dist_parameters=source_parameters, inverse_cdf=source_inverse_cdf)
    obj_instance = obj_init_instance
    obj_init = obj_init_instance.objective_fun()
    obj_list = [obj_init]
    weight_list = [weights_init]
    ite = 0
    while (ite < 2) or (obj_list[ite] - obj_list[ite-1]) > threshold:
        gradient = obj_instance.first_variation()
        weights_now = weights_now - gd_step * gradient
        obj_instance = ObjectivesEval(cost_type=cost_type, kernel_type=kernel_type, alpha=alpha, kernel_params=kernel_params,  weights=weights_now, center_samples=source_sample, dist_type=source_dist_type, mcsample_size=mcsample_size, dist_parameters=source_parameters, inverse_cdf=source_inverse_cdf)
        obj = obj_instance.objective_fun()
        weight_list.append(weights_now)
        obj_list.append(obj)
        ite += 1
        print('ite:', ite, ' obj:', obj)
        print('gradient:', gradient)
    return [np.array(obj_list), np.array(weight_list)]


gd(kernel_dim=10)
import numpy as np
import math
from kernalization import SampleGeneration
from kernalization import FunctionApprox
from evaluation import CostFunction
import matplotlib.pyplot as plt
import pandas as pd


def dual_objective_eval(kernel_mu=None, kernel_nv=None, source_center=None, target_center=None, source_weights=None, target_weights=None, source_dist_type='normal', regularize_parameter=1e-1, source_parameters=np.array([2, 1]), source_inverse_cdf=None, target_dist_type='normal', target_parameters=np.array([0, 1]), target_inverse_cdf=None, source_kernel_type='exponential', target_kernel_type='exponential', kernel_params=None, mcsample_size=100, cost_type='quadratic'):
    source_sample_instance = SampleGeneration(sample_size=mcsample_size, dist_type=source_dist_type,
                                              parameters=source_parameters, inverse_cdf=source_inverse_cdf)
    target_sample_instance = SampleGeneration(sample_size=mcsample_size, dist_type=target_dist_type,
                                              parameters=target_parameters, inverse_cdf=target_inverse_cdf)
    source_sample = source_sample_instance.sampling()
    target_sample = target_sample_instance.sampling()
    source_kernel_instance = FunctionApprox(center_samples=source_center, inputs=source_sample,
                                            weights=source_weights, kernel=kernel_mu,
                                            kernel_type=source_kernel_type, kernel_params=kernel_params)
    target_kernel_instance = FunctionApprox(center_samples=target_center, inputs=target_sample,
                                            weights=target_weights, kernel=kernel_nv,
                                            kernel_type=target_kernel_type, kernel_params=kernel_params)
    u_of_x = source_kernel_instance.predict()
    v_of_y = target_kernel_instance.predict()
    cost_instance = CostFunction(cost_type=cost_type)
    cost = []
    for i in range(len(source_sample)):
        cost.append(cost_instance.cost_eval(source_sample[i], target_sample[i]))
    costs = np.array(cost)
    regularization = regularize_parameter * np.average(np.exp((u_of_x + v_of_y - costs)/regularize_parameter))
    obj = np.average(u_of_x) + np.average(v_of_y) - regularization
    # print(np.average(u_of_x) + np.average(v_of_y), regularization)
    return obj


def dual_rkhs(step_size=0.01, kernel_mu=None, kernel_nv=None, source_dist_type='normal', regularize_parameter=1e-1, source_parameters=np.array([1, 1]), source_inverse_cdf=None, target_dist_type='normal', target_parameters=np.array([0, 1]), target_inverse_cdf=None, source_kernel_type='exponential', target_kernel_type='exponential', alpha=0.5, kernel_params=False, mcsample_size=100, cost_type='quadratic', threshold=1e-10, iterations=3e2):

    res = []
    alpha_list = []
    obj_list = []
    weights_list = []
    ite_list =[]
    x_sample_list = []
    y_sample_list = []
    ite = 1
    # print(obj_list[ite-1] - obj_list[ite-2])
    # while (ite < 2) or (obj_list[ite-1] - obj_list[ite-2]) > threshold:
    while ite <= iterations:
        source_sample_instance = SampleGeneration(sample_size=1, dist_type=source_dist_type,
                                                  parameters=source_parameters, inverse_cdf=source_inverse_cdf)
        target_sample_instance = SampleGeneration(sample_size=1, dist_type=target_dist_type,
                                                  parameters=target_parameters, inverse_cdf=target_inverse_cdf)
        source_sample = source_sample_instance.sampling()
        target_sample = target_sample_instance.sampling()
        x_sample_list.append(source_sample)
        y_sample_list.append(target_sample)
        if ite == 1:
            source_center = source_sample
            target_center = target_sample
            u_k_minus_1 = 0
            v_k_minus_1 = 0
            cost_instance = CostFunction(cost_type=cost_type)
            alpha_k = step_size / math.sqrt(ite) * (1 - math.exp((u_k_minus_1 + v_k_minus_1 - cost_instance.cost_eval(source_sample[0], target_sample[0])) / regularize_parameter))
            alpha_list.append(alpha_k)
            obj_val = dual_objective_eval(kernel_mu=kernel_mu, kernel_nv=kernel_nv, source_center=source_center,
                                          target_center=target_center,
                                          source_weights=np.array([alpha_k]), target_weights=np.array([alpha_k]),
                                          source_dist_type=source_dist_type,
                                          regularize_parameter=regularize_parameter,
                                          source_parameters=source_parameters,
                                          source_inverse_cdf=source_inverse_cdf, target_dist_type=target_dist_type,
                                          target_parameters=target_parameters, target_inverse_cdf=target_inverse_cdf,
                                          source_kernel_type=source_kernel_type, target_kernel_type=target_kernel_type,
                                          kernel_params=kernel_params, mcsample_size=mcsample_size, cost_type=cost_type)
        else:
            source_kernel_instance = FunctionApprox(center_samples=source_center, inputs=source_sample,
                                                    weights=np.array(alpha_list), kernel=kernel_mu,
                                                    kernel_type=source_kernel_type, kernel_params=kernel_params)
            u_k_minus_1 = source_kernel_instance.predict()
            target_kernel_instance = FunctionApprox(center_samples=target_center, inputs=target_sample,
                                                    weights=np.array(alpha_list), kernel=kernel_nv,
                                                    kernel_type=target_kernel_type, kernel_params=kernel_params)
            v_k_minus_1 = target_kernel_instance.predict()
            # print(u_k_minus_1, v_k_minus_1)
            obj_val = dual_objective_eval(kernel_mu=kernel_mu, kernel_nv=kernel_nv, source_center=source_center,
                                          target_center=target_center,
                                          source_weights=np.array(alpha_list),
                                          target_weights=np.array(alpha_list),
                                          source_dist_type=source_dist_type,
                                          regularize_parameter=regularize_parameter,
                                          source_parameters=source_parameters,
                                          source_inverse_cdf=source_inverse_cdf, target_dist_type=target_dist_type,
                                          target_parameters=target_parameters, target_inverse_cdf=target_inverse_cdf,
                                          source_kernel_type=source_kernel_type, target_kernel_type=target_kernel_type,
                                          kernel_params=kernel_params, mcsample_size=mcsample_size, cost_type=cost_type)
            cost_instance = CostFunction(cost_type=cost_type)
            alpha_k = step_size / math.sqrt(ite) * (1 - math.exp((u_k_minus_1 + v_k_minus_1 - cost_instance.cost_eval(
                source_sample[0], target_sample[0])) / regularize_parameter))
            alpha_list.append(alpha_k)
            source_center = np.concatenate((source_center, source_sample))
            target_center = np.concatenate((target_center, target_sample))
            # print(alpha_list, source_center, target_center)
        obj_list.append(obj_val)
        weights_list.append(np.array(alpha_list))
        ite_list.append(ite)
        ite += 1
        # if ite % 1e3 == 0:
        #     print(ite)
        if ite % 1e2 == 0:
            print('iteration number: ', ite)
            print('obj  val: ', obj_val)
        if ite % 1e2 == 0:
            res.append([ite, source_center, target_center, np.array(alpha_list)])
    print(len(ite_list), len(obj_list), len(weights_list))
    d = {'iteration': ite_list, 'objective': obj_list, 'weights': weights_list, 'x_sample': x_sample_list, 'y_sample': y_sample_list}
    df = pd.DataFrame(data=d)
    df.to_csv("data/samplesize="+str(mcsample_size)+"_stepsize="+str(step_size)+"_bandwidth="+str(kernel_params['bandwidth']), index=False)
    return [np.array(obj_list), res, kernel_params['bandwidth'], step_size, df]
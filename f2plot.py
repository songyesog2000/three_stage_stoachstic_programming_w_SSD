import numpy as np
import matplotlib.pyplot as plt

def F2_plot(meta_parameter,source , target):
    n = meta_parameter['number of agents']
    K = meta_parameter['number of resources']

    S = meta_parameter['number of S scenario']
    T = meta_parameter['number of T scenario']
    
    
    for s in range(S):
        prob_res = [source[s][t]['cond_prob'] for t in range(T)]
        value_res = [source[s]['r_s'] + source[s][t]['r_t'] for t in range(T)]
        sort_id_res = np.argsort(value_res).astype(int)
        short_fall_res = [sum(prob_res[tt]*max(0,value_res[t]-value_res[tt]) for tt in range(T)) for t in sort_id_res]
        value_res = [value_res[i] for i in sort_id_res]

        prob_target = [target[s][t]['cond_prob'] for t in range(T)]
        value_target = [target[s]['r_s'] + target[s][t]['r_t'] for t in range(T)]
        sort_id_target = np.argsort(value_target).astype(int)
        short_fall_target = [sum(prob_res[tt]*max(0,value_target[t]-value_target[tt]) for tt in range(T)) for t in sort_id_target]
        value_target = [value_target[i] for i in sort_id_target]
        

        x_min = min(value_target+value_res) - 0.1
        x_max = max(value_res+value_target) + 0.1
        print(sort_id_res)
        plt.figure()
        plt.plot(value_res, short_fall_res, color='tab:blue',label='source') #benchmark
        plt.plot(value_target, short_fall_target, color='tab:orange',label='target')
        plt.legend()

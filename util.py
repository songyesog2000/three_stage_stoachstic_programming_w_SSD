import numpy as np

def F2_policy_stage_3(x_1, x_2, scenario):
    n_item = scenario['n_item']
    S_scenario = scenario['S_scenario']
    L_scenario = scenario['L_scenario']
    
    evaluation = evaluation_policy(x_1, x_2, scenario)
    
    y_3_eval = {}
    for s in range(S_scenario):
        y_3_eval[s] = {evaluation['stage_3'][s][l]['current_tot_reward']:evaluation['stage_3'][s][l]['prob']\
                for l in range(L_scenario)}

    w_eval = {}
    for s in range(S_scenario):
        w_eval[s] = {}
        for y_3_j in y_3_eval[s].keys():
            w_eval[s][y_3_j] = sum(max(y_3_j-y_3_i,0)*prob for y_3_i,prob in y_3_eval[s].items())
    return w_eval

def F2_bcm_stage_3( benchmark, scenario):
    n_item = scenario['n_item']
    S_scenario = scenario['S_scenario']
    L_scenario = scenario['L_scenario']
    
    y_1_bcm = benchmark['stage_1']['r'][0]
    
    y_3_bcm = {}
    for s in range(S_scenario):
        y_3_bcm[s] = {benchmark['stage_3']['r'][s][l]:benchmark['stage_3']['s'][s][l]\
                for l in range(L_scenario)}
        
    w_bcm = {}
    for s in range(S_scenario):
        w_bcm[s] = {}
        for y_3_j in y_3_bcm[s].keys():
            w_bcm[s][y_3_j] = sum(max(y_3_j-y_3_i,0)*prob for y_3_i,prob in y_3_bcm[s].items())
    return w_bcm

def F2_policy_stage_2 (x_1, x_2, scenario):
    n_item = scenario['n_item']
    S_scenario = scenario['S_scenario']
    L_scenario = scenario['L_scenario']
    
    evaluation = evaluation_policy(x_1, x_2, scenario)

    theta = {}
    for s in range(S_scenario):
        theta[s] = evaluation['stage_2'][s]['current_tot_reward'] + \
                        sum(  evaluation['stage_3'][s][l]['revenue']*\
                               evaluation['stage_3'][s][l]['prob'] \
                                                        for l in range(L_scenario))
    u = {}
    for s in range(S_scenario):
        u[theta[s]] =  sum( max(theta[s] -  theta[s_], 0)*evaluation['stage_2'][s_]['prob']\
                           for s_ in range(S_scenario))
        
    return u

def ssd_benchmark_from_function( x_1, x_2_i, scenario):
    
    n_item = scenario['n_item']
    S_scenario = scenario['S_scenario']
    L_scenario = scenario['L_scenario']
    d_2 = scenario['stage_2']['rd_prob'][0][0]
    prob_of_d_2 =  scenario['stage_2']['rd_prob'][0][1]
    
    d_3 = {}
    for s in range(S_scenario):
        d_3[s] = {}
        for l in range(L_scenario):
            d_3[s][l] = scenario['stage_3']['rd_prob'][s][0][l]
    
    prob_of_d_3 = {}
    for s in range(S_scenario):
        prob_of_d_3[s] = {}
        for l in range(L_scenario):
            prob_of_d_3[s][l] = scenario['stage_3']['rd_prob'][s][1][l]
    
    c_1 = scenario['stage_1']['cf']['c_1']
    c_2 = scenario['stage_2']['cf']['c_2']
    p_2 = scenario['stage_2']['cf']['p_2']
    p_3 = scenario['stage_3']['cf']['p_3']
    
    
    benchmark = {}
    benchmark['stage_1'] = {
        'r': [-sum(x_1[i]*c_1[i] for i in range(n_item))],
        's': [1.0]
    }
    
    
    benchmark['stage_2'] = {
        'r': [sum(p_2[i]*min(d_2[s][i],x_1[i])-c_2[i]*x_2_i(x_1[i],d_2[s][i]) for i in range(n_item))\
                          for s in range(S_scenario)],
        's': prob_of_d_2
    }
    
    benchmark['stage_3'] = {
        'r': { 
                s:[sum(p_3[i]*min( d_3[s][l][i], x_2_i(x_1[i],d_2[s][i])) for i in range(n_item)) \
                   for l in range(L_scenario)] for s in range(S_scenario)
        },
        's': { 
                s:prob_of_d_3[s] for s in range(S_scenario)
        }
    }
    
    return benchmark

def ssd_benchmark_from_table( x_1, x_2, scenario):
    
    n_item = scenario['n_item']
    S_scenario = scenario['S_scenario']
    L_scenario = scenario['L_scenario']
    d_2 = scenario['stage_2']['rd_prob'][0][0]
    prob_of_d_2 =  scenario['stage_2']['rd_prob'][0][1]
    
    d_3 = {}
    for s in range(S_scenario):
        d_3[s] = {}
        for l in range(L_scenario):
            d_3[s][l] = scenario['stage_3']['rd_prob'][s][0][l]
    
    prob_of_d_3 = {}
    for s in range(S_scenario):
        prob_of_d_3[s] = {}
        for l in range(L_scenario):
            prob_of_d_3[s][l] = scenario['stage_3']['rd_prob'][s][1][l]
    
    c_1 = scenario['stage_1']['cf']['c_1']
    c_2 = scenario['stage_2']['cf']['c_2']
    p_2 = scenario['stage_2']['cf']['p_2']
    p_3 = scenario['stage_3']['cf']['p_3']
    
    
    benchmark = {}
    benchmark['stage_1'] = {
        'r': [-sum(x_1[i]*c_1[i] for i in range(n_item))],
        's': [1.0]
    }
    
    
    benchmark['stage_2'] = {
        'r': [sum(p_2[i]*min(d_2[s][i],x_1[i])-c_2[i]*x_2[s][i] for i in range(n_item))\
                          for s in range(S_scenario)],
        's': prob_of_d_2
    }
    
    benchmark['stage_3'] = { 
        'r': { 
                s:[sum(p_3[i]*min( d_3[s][l][i], x_2[s][i]) for i in range(n_item)) \
                   for l in range(L_scenario)] for s in range(S_scenario)
        },
        's': { 
                s:prob_of_d_3[s] for s in range(S_scenario)
        }
    }
    
    return benchmark
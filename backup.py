import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
from gurobipy import quicksum as qsum

def three_stage_problem_no_SSD(scenario, return_solution=False):
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
    
    m = gp.Model(f'main')
    m.Params.LogToConsole = 0
    x_1 = m.addVars(n_item, vtype = GRB.CONTINUOUS, name = 'x_1')
    x_2 = {}
    for s in range(S_scenario):
        x_2[s] = m.addVars(n_item, vtype = GRB.CONTINUOUS, name = f'x_2_s={s}')
    
    # retail quantatity 
    # gurobi itself provide genral constraint to handle max or min functions  
    # which introduces extra binary variables into the model
    s_2 = {}
    for i in range(S_scenario):
        s_2[i] = m.addVars(n_item, vtype = GRB.CONTINUOUS, name = f's_2_s={i}')
        for j in range(n_item):
            m.addConstr( s_2[i][j] == gp.min_(x_1[j], constant = d_2[i][j]))
    
    # budget constraint
    cost_1 = qsum(x_1[i]*c_1[i] for i in range(n_item))
    m.addConstr(cost_1<=scenario['stage_1']['cf']['b_1'], name='budget_constr_stage_1')
    
    budget_rest_1 = scenario['stage_1']['cf']['b_1'] - cost_1
    
    tot_2_l = {}
    tot_2_r = {}
    budget_rest_2 = {}
    for i in range(S_scenario):
        tot_2_r[i] = qsum(p_2[j]*s_2[i][j] for j in range(n_item))
        tot_2_l[i] = qsum(x_2[i][j]*c_2[j] for j in range(n_item))
        m.addConstr(tot_2_l[i]<=tot_2_r[i]+budget_rest_1+scenario['stage_2']['cf']['b_2'], \
                  name='budget_constr_stage_2')
        budget_rest_2[i] = tot_2_r[i]+budget_rest_1+scenario['stage_2']['cf']['b_2'] - tot_2_l[i]
    # revenue
    revenue_2_expected = qsum(tot_2_r[i]*prob_of_d_2[i] for i in range(S_scenario))
    cost_2_expected = qsum(tot_2_l[i]*prob_of_d_2[i] for i in range(S_scenario))
    
    revenue_3_conditional_expected = {}
    s_3 = {}
    for i in range(S_scenario):
        revenue_3 = {}
        excess_3 = {}
        s_3[i] = {}
        for l in range(L_scenario):
            revenue_3[l] = 0
            excess_3[l] = m.addVars(n_item, \
                                   vtype = GRB.CONTINUOUS, \
                                   name = f'excess_3_{i}_{l}')
            s_3[i][l] = m.addVars(n_item, \
                                   vtype = GRB.CONTINUOUS, \
                                   name = f's_3_{i}_{l}')
            for j in range(n_item):
                m.addConstr( excess_3[l][j] == x_2[i][j]+x_1[j]-s_2[i][j])
                m.addConstr( s_3[i][l][j] == gp.min_(excess_3[l][j],constant = d_3[i][l][j]))
                revenue_3[l] += p_3[j]*s_3[i][l][j]
        revenue_3_conditional_expected[i] = qsum(prob_of_d_3[i][l]*revenue_3[l]\
                                                 for l in range(L_scenario))+budget_rest_2[i]
    revenue_3_expected = qsum(revenue_3_conditional_expected[i]*prob_of_d_2[i]\
                              for i in range(S_scenario))
    
    # since budget is constrained
    # maximize revenue is equal to maximize the profit
    obj = revenue_2_expected - cost_1 + revenue_3_expected-cost_2_expected
    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()

   
    if return_solution:
        return [_.X for _ in x_1.values()], \
                [[_.X for _ in x_2[k].values() ] for k in range(S_scenario)]
    else:
        print(cost_1.getValue())
        print(budget_rest_1.getValue())
        print([tot_2_l[i].getValue() for i in range(S_scenario)])
        print([tot_2_r[i].getValue() for i in range(S_scenario)])
        print([budget_rest_2[i].getValue() for i in range(S_scenario)])
        for s in range(S_scenario):
            print(s, [sum(s_3[s][l][j].X*p_3[j] for j in range(n_item)) for l in range(L_scenario)])
        print(obj.getValue())
        return m
        
def evaluation_policy(x_1, x_2, scenario):
    n_item = scenario['n_item']
    S_scenario = scenario['S_scenario']
    L_scenario = scenario['L_scenario']
    d_2 = scenario['stage_2']['rd_prob'][0][0]
    prob_of_d_2 =  scenario['stage_2']['rd_prob'][0][1]
    
    d_3 = {}
    for s in range(S_scenario):
        d_3[s] = scenario['stage_3']['rd_prob'][s][0]
    
    prob_of_d_3 = {}
    for s in range(S_scenario):
        prob_of_d_3[s] = scenario['stage_3']['rd_prob'][s][1]
    
    c_1 = scenario['stage_1']['cf']['c_1']
    c_2 = scenario['stage_2']['cf']['c_2']
    p_2 = scenario['stage_2']['cf']['p_2']
    p_3 = scenario['stage_3']['cf']['p_3']
    b_2 = scenario['stage_2']['cf']['b_2']
    
    rst = {}
    rst['stage_1']={
        'revenue': 0,
        'cost':sum(c_1[j]*x_1[j] for j in range(n_item)),
        'current_tot_reward':0 - sum(c_1[j]*x_1[j] for j in range(n_item)),
        'prob': 1.0
    }
    rst['stage_2'] = {}
    rst['stage_3'] = {}
    
    s_2 = {}
    
    for s in range(S_scenario):
        
        s_2[s] = [min(x_1[j], d_2[s][j]) for j in range(n_item)]
        
        rst['stage_2'][s]={
            'revenue': sum(p_2[j]*s_2[s][j] for j in range(n_item)),
            'cost':sum(c_2[j]*x_2[s][j] for j in range(n_item)),
            'current_tot_reward':rst['stage_1']['current_tot_reward']+\
                                    sum(p_2[j]*min(x_1[j], d_2[s][j]) for j in range(n_item))\
                                    - sum(c_2[j]*x_2[s][j] for j in range(n_item)),
            'prob':prob_of_d_2[s]
        }
        rst['stage_3'][s] = {}
        for l in range(L_scenario):
            rst['stage_3'][s][l] = {
                'revenue': sum(p_3[j]*min(x_1[j]+x_2[s][j]-s_2[s][j], d_3[s][l][j]) for j in range(n_item)),
                'cost':0,
                'current_tot_reward':rst['stage_2'][s]['current_tot_reward']+\
                                    sum(p_3[j]*min(x_1[j]+x_2[s][j]-s_2[s][j], d_3[s][l][j]) for j in range(n_item)),
                'prob':prob_of_d_3[s][l]
            }
    return rst


def ssd_benchmark_from_evaluation( evaluation, scenario):
    
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
        'r': [-evaluation['stage_1']['cost']],
        's': [1.0]
    }
    
    
    benchmark['stage_2'] = {
        'r': [evaluation['stage_2'][s]['revenue']-evaluation['stage_2'][s]['cost'] for s in range(S_scenario)],
        's': prob_of_d_2
    }
    
    benchmark['stage_3'] = {
        'r': { 
                s:[ evaluation['stage_3'][s][l]['revenue']-evaluation['stage_3'][s][l]['cost'] \
                   for l in range(L_scenario)] for s in range(S_scenario)
        },
        's': { 
                s:prob_of_d_3[s] for s in range(S_scenario)
        }
    }
    
    return benchmark



def reward_problem_of_first_stage_ssd(x_1:list, z_1:float, s:int, \
                                      d_2:list,\
                                      d_3:dict,prob_of_d_3:list, 
                                      benchmark:dict,\
                                      w, \
                                      b_2, p_2, p_3, c_2):
    L_scenario = len(prob_of_d_3)
    n_item = len(p_3)
    
    itr = 0
    max_itr = 100
    
    m = gp.Model('reward_problem_two_stage_multi_item_SSD')
    m.Params.LogToConsole = 0
    
    s_2 = m.addVars(n_item, vtype = GRB.CONTINUOUS, name = 's_2')
    x_2 = m.addVars(n_item, vtype = GRB.CONTINUOUS, name = 'x_2')
    z_2 = m.addVar(lb = -float('inf'), ub = float('inf'),\
                       vtype = GRB.CONTINUOUS, name = 'z_2')
    sigma = m.addVar(lb = -float('inf'), ub = float('inf'),\
                       vtype = GRB.CONTINUOUS, name = 'sigma')
    
    pi_1 = m.addConstrs((s_2[j] <= x_1[j] for j in range(n_item)))
    m.addConstrs((s_2[j] <= d_2[j] for j in range(n_item)))
        
    f_3 = {}
    for i in range(L_scenario):
        f_3_tem = m.addVars(n_item,vtype = GRB.CONTINUOUS, name = f'f_3_l={i}')
        m.addConstrs((f_3_tem[j] <= p_3[j]*d_3[i][j] for j in range(n_item)))
        m.addConstrs((f_3_tem[j] <= p_3[j]*x_2[j] for j in range(n_item)))
        f_3[i] = qsum( f_3_tem[j] for j in range(n_item))
    obj = z_2+ qsum(f_3[l]*prob_of_d_3[l] for l in range(L_scenario))
    m.setObjective(obj,GRB.MAXIMIZE)
    
    tot_cost_2 = qsum( c_2[i]*x_2[i] for i in range(n_item))
    tot_revenue_2 = sum(s_2[i]*p_2[i] for i in range(n_item))
    m.addConstr( tot_cost_2 <=\
                          tot_revenue_2 + b_2, name='budget_limit' )
    m.addConstr( z_2<= tot_revenue_2-tot_cost_2, 'z_2<=f(x)')
    
    pi_2 = m.addConstr(sigma ==z_1 + z_2 - \
                       benchmark['stage_1']['r'][0] - \
                       benchmark['stage_2']['r'][s] , name = 'sigma')
    
    # union hte same rewards in benchmark

    f_3_ = {}
    for i in range(L_scenario):
        f_3_[i] = - float('inf')
    x_2_ = [0 for i in range(n_item)]
    z_2_ = 0
        
    while( itr< max_itr):
        m.optimize()
#         print(m.display())
        if m.Status!=2:
            print(m, 'status:',m.Status)
            return  None
        for i in range(n_item):
            x_2_[i] = x_2[i].X
        z_2_ = z_2.X
        sigma_ = sigma.X
        for i in range(L_scenario):
            f_3_[i] = f_3[i].getValue()
        obj_ = obj.getValue()
        
        all_sat_flag = True
       
        for y_3_j,w_j in w.items():
            # print(f'max_{itr}', y_3_j, sigma_,[ f_3_[l] for l in range(L_scenario)], w_j)
            if sum(max(y_3_j-sigma_-f_3_[l],0)*prob_of_d_3[l] for l in range(L_scenario))>w_j:
                all_sat_flag = False
                A = [l for l in range(L_scenario) if y_3_j>sigma_+f_3_[l]]
                a = qsum( (y_3_j-sigma-f_3[d])*prob_of_d_3[d] for d in A)
                m.addConstr( a<=w_j, name= f'event_cut_{itr}_{y_3_j}_{w_j}')

        if itr!=0 and all_sat_flag:
            pi_1_=[pi_1[i].Pi for i in range(n_item)]
            pi_2_=pi_2.Pi
            # print(itr)
#             print('z_2', z_2_)
#             print('sigma',sigma_)
#             print('f_3', f_3_)
            return x_2_,z_2_,obj_,pi_1_,pi_2_, f_3_
        itr +=1
    return None


def feasibility_problem_of_first_stage_ssd(x_1:list, z_1:float, s:int, \
                                      d_2:list,\
                                      d_3:dict,prob_of_d_3:dict, 
                                      benchmark:dict,\
                                      w, \
                                      b_2, p_2, p_3, c_2):
    L_scenario = len(prob_of_d_3)
    n_item = len(p_3)
    
    itr = 0
    max_itr = 100
    
    m = gp.Model('feasiblity_two_stage_SSD')
    m.Params.LogToConsole = 0
    
    s_2 = m.addVars(n_item, vtype = GRB.CONTINUOUS, name = 's_2')
    x_2 = m.addVars(n_item, vtype = GRB.CONTINUOUS, name = 'x_2')
    z_2 = m.addVar(lb = -float('inf'), ub = float('inf'),\
                       vtype = GRB.CONTINUOUS, name = 'z_2')
    sigma = m.addVar(lb = -float('inf'), ub = float('inf'),\
                       vtype = GRB.CONTINUOUS, name = 'sigma')
    u_1 =m.addVars(n_item,lb = -float('inf'), ub = float('inf'), \
                       vtype = GRB.CONTINUOUS, name = 'u_1')
    u_2 = m.addVar(lb = -float('inf'), ub = float('inf'),\
                       vtype = GRB.CONTINUOUS, name = 'u_2')
    u_1_abs = m.addVars(n_item,vtype = GRB.CONTINUOUS, name = 'u_1_abs')
    u_2_abs = m.addVar(vtype = GRB.CONTINUOUS, name = 'u_2_abs')
    
    m.addConstrs( (u_1_abs[i] >= u_1[i] for i in range(n_item)))
    m.addConstrs( (u_1_abs[i] >= -u_1[i] for i in range(n_item)))
    m.addConstr( u_2_abs >= u_2)
    m.addConstr( u_2_abs >= -u_2)
    
                
    
    f_3 = {}
    for i in range(L_scenario):
        f_3_tem = m.addVars(n_item,vtype = GRB.CONTINUOUS, name = f'f_3_l={i}')
        m.addConstrs((f_3_tem[j] <= p_3[j]*d_3[i][j] for j in range(n_item)))
        m.addConstrs((f_3_tem[j] <= p_3[j]*x_2[j] for j in range(n_item)))
        f_3[i] = qsum( f_3_tem[j] for j in range(n_item))
    
    obj = qsum(u_1_abs[i] for i in range(n_item)) + u_2_abs
    m.setObjective(obj,GRB.MINIMIZE)
    
    pi_1 = m.addConstrs((s_2[j]+u_1[j] <= x_1[j] for j in range(n_item)))
    m.addConstrs((s_2[j] <= d_2[j] for j in range(n_item)))
    tot_cost_2 = qsum( c_2[i]*x_2[i] for i in range(n_item))
    tot_revenue_2 = sum(s_2[i]*p_2[i] for i in range(n_item))
    m.addConstr( tot_cost_2 <=\
                          tot_revenue_2 + b_2, name='budget_limit' )
    m.addConstr( z_2+ u_2 <= tot_revenue_2-tot_cost_2, 'z_2<=f(x)')
    
    pi_2 = m.addConstr(sigma ==z_1 + z_2 - \
                       benchmark['stage_1']['r'][0] - \
                       benchmark['stage_2']['r'][s] , name = 'sigma')
    
    # union hte same rewards in benchmark

    f_3_ = {}
    for i in range(L_scenario):
        f_3_[i] = - float('inf')
    x_2_ = [0 for i in range(n_item)]
    z_2_ = 0
        
    while( itr< max_itr):
        m.optimize()
        # print(m.display())
        if m.Status!=2:
            print(m, 'status:',m.Status)
            return  None
        for i in range(n_item):
            x_2_[i] = x_2[i].X
        z_2_ = z_2.X
        sigma_ = sigma.X
        for i in range(L_scenario):
            f_3_[i] = f_3[i].getValue()
        obj_ = obj.getValue()
        
        all_sat_flag = True
        for y_3_j,w_j in w.items():
            if sum(max(y_3_j-sigma_-f_3_[l],0)*prob_of_d_3[l] for l in range(L_scenario))>w_j+0.01:
                all_sat_flag = False
                A = [l for l in range(L_scenario) if y_3_j>sigma_+f_3_[l]]
                a = qsum( (y_3_j-sigma-f_3[d])*prob_of_d_3[d] for d in A)
                m.addConstr( a<=w_j, name= f'event_cut_{itr}_{y_3_j}_{w_j}')

        if itr!=0 and all_sat_flag:
            pi_1_=[pi_1[i].Pi for i in range(n_item)]
            pi_2_=pi_2.Pi
            # print(f'iteration times: {itr+1}')
            
            return x_2_,z_2_, obj_,pi_1_,pi_2_
        itr +=1
    
    return 'beyond max_itr'

def three_stage_multi_item_SSD_newsvendor(x_1_0, z_1_0, scenario, benchmark, max_itr = 100,\
                                         policy_return = False):
    obs_rst = {}
    n_item = scenario['n_item']
    S_scenario = scenario['S_scenario']
    L_scenario = scenario['L_scenario']
    d_2 = scenario['stage_2']['rd_prob'][0][0]
    prob_of_d_2 =  scenario['stage_2']['rd_prob'][0][1]
    
    d_3 = {}
    for s in range(S_scenario):
        d_3[s] = scenario['stage_3']['rd_prob'][s][0]
    
    prob_of_d_3 = {}
    for s in range(S_scenario):
        prob_of_d_3[s] = scenario['stage_3']['rd_prob'][s][1]
    
    c_1 = scenario['stage_1']['cf']['c_1']
    c_2 = scenario['stage_2']['cf']['c_2']
    p_2 = scenario['stage_2']['cf']['p_2']
    p_3 = scenario['stage_3']['cf']['p_3']
    b_2 = scenario['stage_2']['cf']['b_2']
    
    y_1 = benchmark['stage_1']['r'][0]
    
    y = {}
    for s in range(S_scenario):
        y[s] = {benchmark['stage_3']['r'][s][l]:benchmark['stage_3']['s'][s][l]\
                for l in range(L_scenario)}
        
    w = {}
    for s in range(S_scenario):
        w[s] = {}
        for y_3_j in y[s].keys():
            w[s][y_3_j] = sum(max(y_3_j-y_3_i,0)*prob for y_3_i,prob in y[s].items())
        
    theta = {}
    for s in range(S_scenario):
        theta[s] = benchmark['stage_2']['r'][s] + \
                        sum(  y_3_i*prob for y_3_i,prob in y[s].items())
    u = {}
    for s in range(S_scenario):
        u[s] =  sum( max(theta[s] -  theta[s_], 0)*prob_of_d_2[s_] for s_ in range(S_scenario))
        
    # initialize, step 0
    itr = 0
    
    event_n = 0
    event_list = []
    event_cut = {}
    obj_cut = []
    fs_cut = []
    
    x_1_ = x_1_0
    z_1_ = z_1_0
    
    v_ = {}
    for i in range(S_scenario):
        v_[i]= float('inf')
    
    # init the master problem
    # we set the gurobi model outside the iteration loop
    master = gp.Model('master problem')
    
    # master.Params.LogToConsole = 0
    x_1 = master.addVars(n_item,lb=0,ub=10, vtype = GRB.CONTINUOUS, name = 'x_1')
    z_1 = master.addVar(lb = -float('inf'), ub = float('inf'),\
                            vtype = GRB.CONTINUOUS, name = 'z_1')
    
    # x_1 and z_1
    cost_tot_1 = qsum( c_1[i]*x_1[i] for i in range(n_item))
    master.addConstr(z_1+cost_tot_1<=0)
    
    # variable v for each second stage scenario
    v = {}
    for i in range(n_item):
        v[i] = master.addVar(lb = -float('inf'), ub = 1000,\
                               vtype = GRB.CONTINUOUS, name = f'v_s={i}')

    
    master_obj = z_1 + sum( v[i]*prob_of_d_2[i] for i in range(S_scenario) )
    master.setObjective(master_obj, GRB.MAXIMIZE)
    
    
    reward_second_stage = [float('inf') for s in range(S_scenario)]
    x_2_ = [[0 for i in range(n_item)] for s in range(S_scenario)]
    # cut inequalities are added into master during iterations
    while(itr<max_itr):
        
        all_solvable_flag = True
        for s in range(S_scenario):
            rst_obj = reward_problem_of_first_stage_ssd(x_1_, z_1_, s, \
                                                  d_2[s],
                                                  d_3[s], prob_of_d_3[s], \
                                                  benchmark,\
                                                  w[s], \
                                                  b_2, p_2, p_3, c_2)
            
            if rst_obj is not None:
                # try objective cuts
                x_2_[s], _, reward_second_stage[s], pi_1_,pi_2_, obs_rst[s] = rst_obj
                # print(s, rst_obj)
                tem_obj_cut = master.addConstr( v[s] <=reward_second_stage[s] -\
                                               qsum(pi_1_[i]*(x_1[i]-x_1_[i])\
                                                                for i in range(n_item))\
                                                          + pi_2_*(z_1 - z_1_),\
                                              name= f'obj_cut_{itr}_{s}')
                obj_cut.append(tem_obj_cut)
            else:
                reward_second_stage[s] = -float('inf')
                # try feasibility cuts
                rst_fs = feasibility_problem_of_first_stage_ssd(x_1_, z_1_, s, \
                                                  d_2[s],
                                                  d_3[s], prob_of_d_3[s], \
                                                  benchmark,\
                                                  w[s], \
                                                  b_2, p_2, p_3, c_2)
                # print(rst_fs)
                if rst_fs=='optimal': 
                    continue
                elif rst_fs=='beyond max_itr':
                    raise RuntimeError(rst_fs + f' in {itr} scenario = {s}')
                all_solvable_flag = False
                _, _, fs_obj, pi_1_,pi_2_ = rst_fs
                tem_fs_cut = master.addConstr(  0 >= fs_obj -\
                                             qsum(pi_1_[i]*(x_1[i]-x_1_[i])\
                                                                for i in range(n_item))\
                                                     + pi_2_*(z_1 - z_1_), \
                                             name= f'fs_cut_{itr}_{s}')
                fs_cut.append(tem_fs_cut)
        
        master.optimize()
        
        # print(master.Status)
        if master.Status==4:
            return 'unbounded'
        
        # update values of variables
        x_1_ = [x_1[i].X for i in range(n_item)] 
        z_1_=round2(z_1.X)
        for s in range(S_scenario):
            v_[s]= round2(v[s].X)
        print(itr,'x_1: ', x_1_,'z_1: ', z_1_)
        
        # update (add) event cuts to master problem
        event_n_ = event_n 
        if all_solvable_flag:
            A = {}
            tem_sup = {}
            
            for j in range(S_scenario):
                # print( z_1_, y_1, theta[j], [v_[i] for i in range(S_scenario)], u[j])
                A[j] = [i for i in range(S_scenario) if v_[i] + z_1_-y_1<=theta[j]]
                
                print(A[j])
                tem_sup[j] = sum((theta[j] - v_[i]-z_1_+y_1)*prob_of_d_2[i]\
                                       for i in A[j]) - u[j]
                
            tem_sup_ = max(sup for key,sup in tem_sup.items())
            if tem_sup_ > 0:
                tem_sup_ = tem_sup_/2
                for s in range(S_scenario):
                    if tem_sup[s]>=tem_sup_:
                        # add new constraints
                        tot = 0
                        for j in A[s]:
                            tem_lhs = theta[s] -v[j] -z_1+y_1
                            tot += prob_of_d_2[j]*tem_lhs
                        master.addConstr(tot<=u[s],
                                        name= f'event_cut_{itr}_{s}')
                        event_n +=1        
        
        tolerence = max(abs(v_[s]- reward_second_stage[s])\
                                     for s in range(S_scenario))
        # print(tolerence)
        # print('abs',[(v_[s], reward_second_stage[s]) for s in range(S_scenario)])
        if event_n==event_n_ and all_solvable_flag\
                and tolerence<0.01:
            break
        
        # increase k by 1
        itr+=1
    print(itr,'iterations')
    print(master.display())
    # print(obs_rst)
    if (itr>=max_itr):
        print('abs',[(v_[s], reward_second_stage[s]) for s in range(S_scenario)])
        print("terminated: max iteration")
        return x_1_,x_2_
    
    elif policy_return:
        return x_1_,x_2_
    else:
        return x_1_, z_1_
##### 
risk_neutral_rst = evaluation_policy(x_1_opt, x_2_opt, scenario)
bcm_rst = evaluation_policy(x_1_dummy, x_2_dummy, scenario)
dummy_benchmark = ssd_benchmark_from_evaluation(bcm_rst, scenario)
risk_neutral_benchmark = ssd_benchmark_from_evaluation(risk_neutral_rst, scenario)
x_1,x_2 = three_stage_multi_item_SSD_newsvendor(x_1_0 = x_1_0, z_1_0 =z_1_0, scenario=scenario, \
                                benchmark = dummy_benchmark , max_itr = 20,\
                                policy_return =  True)
dummy_ssd_rst = evaluation_policy(x_1, x_2, scenario)


eval_rst = dummy_ssd_rst
expected = 0
for s in range(S_scenario):
    expected += eval_rst['stage_2'][s]['prob']*\
        sum(eval_rst['stage_3'][s][l]['current_tot_reward']*eval_rst['stage_3'][s][l]['prob']\
            for l in range(L_scenario))
print(expected)

eval_rst = bcm_rst
expected = 0
for s in range(S_scenario):
    expected += eval_rst['stage_2'][s]['prob']*\
        sum(eval_rst['stage_3'][s][l]['current_tot_reward']*eval_rst['stage_3'][s][l]['prob']\
            for l in range(L_scenario))
print(expected)

def x_1_dummy_generator(scenario):
    n_item = scenario['n_item']
    c_1 = scenario['stage_1']['cf']['c_1']
    b_1 = scenario['stage_1']['cf']['b_1']
    mul = b_1/n_item
    return [mul/c_1[i] for i in range(n_item)]

x_1_dummy = x_1_generator(scenario)
x_2_dummy = [[x_2_i(x_1[j],d_2[j]/2) for j in range(n_item)] \
       for d_2 in scenario['stage_2']['rd_prob'][0][0] ]

risk_neutral_rst = evaluation_policy(x_1_opt, x_2_opt, scenario)
bcm_rst = evaluation_policy(x_1_dummy, x_2_dummy, scenario)
dummy_benchmark = ssd_benchmark_from_evaluation(bcm_rst, scenario)
risk_neutral_benchmark = ssd_benchmark_from_evaluation(risk_neutral_rst, scenario)
x_1,x_2 = three_stage_multi_item_SSD_newsvendor(x_1_0 = x_1_0, z_1_0 =z_1_0, scenario=scenario, \
                                benchmark = dummy_benchmark , max_itr = 20,\
                                policy_return =  True)
dummy_ssd_rst = evaluation_policy(x_1, x_2, scenario)

x_1_eval,x_2_eval = x_1,x_2
x_1_bcm, x_2_bcm = x_1_dummy,x_2_dummy

import matplotlib.pyplot as plt
fig = plt.gcf()
fig.set_size_inches(15, 12)
ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)
axs = [ax1,ax2,ax3,ax4,ax5]
w_eval = F2_policy_stage_3(x_1,x_2,scenario)
w_bcm = F2_policy_stage_3(x_1_dummy, x_2_dummy, scenario)
plt.xlabel('x') 
for i in range(scenario['S_scenario']):
    sort_keys = sorted(w_eval[i].keys())
    sort_keys_bcm = sorted(w_bcm[i].keys())
    end = max(sort_keys[-1],sort_keys_bcm[-1])
    w_bcm[i][end] = (end-sort_keys_bcm[-1])+w_bcm[i][sort_keys_bcm[-1]]

    sort_keys_bcm.append(end)
    axs[i].plot(sort_keys,[w_eval[i][_] for _ in sort_keys], label='SSD policy', linewidth=4)
    axs[i].plot(sort_keys_bcm,[w_bcm[i][_] for _ in sort_keys_bcm], label='benchmark', linewidth=4)
plt.legend()
plt.show()

fig = plt.gcf()
fig.set_size_inches(10, 8)
x_1_eval,x_2_eval = x_1,x_2
x_1_bcm, x_2_bcm = x_1_dummy,x_2_dummy
u_eval = F2_policy_stage_2(x_1_eval,x_2_eval,scenario)
u_bcm = F2_policy_stage_2(x_1_bcm, x_2_bcm, scenario)
sort_keys = sorted(u_eval.keys())
sort_keys_bcm = sorted(u_bcm.keys())
end = max(sort_keys[-1],sort_keys_bcm[-1])
u_bcm[end] = (end-sort_keys_bcm[-1])+u_bcm[sort_keys_bcm[-1]]
sort_keys_bcm.append(end)
plt.plot(sort_keys,[u_eval[_] for _ in sort_keys], label='SSD policy',linewidth=4)
plt.plot(sort_keys_bcm,[u_bcm[_] for _ in sort_keys_bcm], label='benchmark',linewidth=4)
plt.xlabel('x',size = 20)
plt.ylabel('F^2 values', size = 20)
plt.legend()
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.show()


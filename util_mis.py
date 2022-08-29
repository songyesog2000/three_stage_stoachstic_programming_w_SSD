import random
import pickle

def generate_scenario(meta_parameter):
    #S scenarios in second stage, T in extended stage
    n = meta_parameter['number of agents']

    S = meta_parameter['number of S scenario']
    T = meta_parameter['number of T scenario']
    scenario = {}

    pT=np.random.rand(S,T)
    pT=pT/sum(sum(pT))
    pT=pT.tolist()
    
    for j in range(S):
        scenario[j] = {}
        xS=[ random.uniform(0,0.6) for i in range(n)]
        scenario[j]['xi_1'] =  xS
        scenario[j]['prob'] = sum(pT[j])
        
        for k in range(T):
            xT=[ random.uniform(0,1) for i in range(n)]
            scenario[j][k]={'xi_2':xT,'prob':pT[j][k],'cond_prob':pT[j][k]/sum(pT[j])}
    return scenario 

    
def save_data(file_name, meta_parameter, scenario):
    with open(file_name, 'wb') as handle:
        pickle.dump(meta_parameter, handle)
        pickle.dump(scenario, handle)

def read_data(file_name):
    with open(file_name, 'rb') as handle:
        meta_parameter = pickle.load(handle)
        scenario = pickle.load(handle)
    return meta_parameter,scenario
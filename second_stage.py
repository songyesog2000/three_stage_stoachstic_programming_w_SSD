import math
import numpy as np
import math
import numpy as np
from cvxopt import solvers,

max_itr = 100000
solver_list = ['CVXOPT']
programming_list = ['cp','cpl','linear']
class second_stage_solver:
    
    def __init__(self, problem_data, solver_type=['cp','CVXOPT']) -> None:
        
        self.events = []
    
    def _get_events(self, x_2) -> list:
        L = self.scenarios_num
        rewards = self.reward(x_2)
        events = []
        s = self.sigma
        for j,(y_j, w_j) in enumerate(zip(self.y_3,self.w)): # this for can be parallelism
            if self.p*((y_j - s)*np.ones(self.n) - rewards) <= w_j:
                continue
            event = [ i for i in range(L) if y_j -s > rewards[i]] 
            events.append(event)

    def solve(self):
        if self.solver_type[0] in programming_list and self.solver_type[1] in solver_list:
            raise ValueError(f'{self.solver_type} is not available')
        if self.solver_type[1]=='CVXOPT' and self.solver_type[0]=='cp':
            F = self.cvxopt_F(self.reward)
            if isinstance(F,str):
                raise ValueError(F)
            constraints = self.cvxopt_constraints(self.constraints)
            # use self.function(self.data) for the clarity
            # it can be optimized as self.function()
            if isinstance(constraints,str):
                raise ValueError(constraints)
            self.init_constraint = constraints

            # start the multicut process:
            itr =0
            while(itr<max_itr):
                A,b = self.cvxopt_update('event_cut')
                # user should use a custom cvxopt_update function regarding to their problems
                solvers.cp(F, A=A, b=b)

    def cvxopt_update(self, problem = None):
        if not problem:
            self.init_constraint
        if problem == 'event_cut':
            self.events.extend(self._get_events())


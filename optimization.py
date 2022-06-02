import numpy as np
import random
import math
from typing import List

class Optimizer:
    def __init__(self, domains, cost_fn, grid_steps):
        self.domains = domains
        self.cost_fn = cost_fn
        self.grid_steps = grid_steps
        self.neighbors_fn = Optimizer.grid_neighbors
        self.best_sol:np.ndarray = None
        self.has_cooled = False
        
    @staticmethod
    def grid_neighbors(domains, sol, grid_steps) -> List:
        """ Return the neighbors solutions in the grid """
        dimensions = len(sol)
        neighbors = []
        for d in range(dimensions):
            if (sol[d]+grid_steps[d] <= domains[d][1]):
                upper_sol = sol.copy()
                upper_sol[d] = sol[d]+grid_steps[d]
                neighbors.append(upper_sol)
            elif d>0: # Circular search only for the angles
                upper_sol = sol.copy()
                upper_sol[d] = domains[d][0]
                neighbors.append(upper_sol)
            if (sol[d]-grid_steps[d] >= domains[d][0]):
                lower_sol = sol.copy()
                lower_sol[d] = sol[d]-grid_steps[d]
                neighbors.append(lower_sol)
            elif d>0: # Circular search only for the angles
                lower_sol = sol.copy()
                lower_sol[d] = domains[d][1]
                neighbors.append(lower_sol)
        return neighbors
    
    def step(self) -> bool:
        """ Executes one optimization iteration """
        """ Returns True/False indicating if a better solution have been found during this iteration """
        """         If True the solution has to be draw """
        """         In this case best_sol, best_E/best_cost contain the best solution found """
        raise NotImplementedError('abstract method')
    
    def applyCooling(self) -> None: 
        assert self.has_cooled == False, 'Cooling is applied only once'
        for i in range(len(self.grid_steps)):
            self.grid_steps[i] = self.grid_steps[i] / 5 # Cooling
        self.current_sol = self.best_sol
        self.has_cooled = True   

class SAOptimizer (Optimizer):
    def __init__(self, domains, cost_fn, grid_steps):
        super().__init__(domains, cost_fn, grid_steps)
        
    def restart(self, start_sol, T=10000.0, cool_factor = 0.99, stopT = 1.0) -> None:
        self.T = T
        self.cool_factor = cool_factor
        self.stopT = stopT
        self.best_sol = self.current_sol = np.array(start_sol)
        self.best_E  = self.cost_fn(start_sol)
        
    def step(self) -> bool:
        assert self.T is not None, 'Call restart first'
        assert not self.hasFinished(), 'You should not call step() if the optimization had finished'
        Ea = self.cost_fn(self.current_sol)
        # Choice a random neighbor sol
        neighbors = self.neighbors_fn(self.domains, self.current_sol, self.grid_steps)
        next_sol = random.choice(neighbors)
        # Calculate next energy
        Eb = self.cost_fn(next_sol)
        # Update sol if next_sol has lower cost (p>1)
        # or we pass the probability cutoff
        better_sol_found = False
        p = math.pow(math.e, (Ea-Eb)/self.T)
        if (np.random.uniform() < p):
            self.current_sol = next_sol
            Ea = Eb
            # Save the best ever found
            if (Eb < self.best_E):
                self.best_sol = next_sol
                self.best_E = Eb
                better_sol_found = True
        elif (not self.has_cooled):
            self.applyCooling()
        # Decrease temperature
        self.T = self.cool_factor * self.T
        return better_sol_found

    def hasFinished(self):
        return self.T<self.stopT

class TSOptimizer (Optimizer):
    def __init__(self, domains, cost_fn, grid_steps):
        grid_steps = grid_steps 
        super().__init__(domains, cost_fn, grid_steps)
    
    def restart(self, start_sol, max_it=1000, max_tl_len = 50) -> None:
        self.pending_it = max_it
        self.cooling_it = int(0.20*max_it) 
        self.max_tl_len = 100
        self.tl = [start_sol]
        self.best_sol = self.current_sol = np.array(start_sol)
        self.best_cost  = self.cost_fn(start_sol)

    def step(self) -> bool:
        assert self.pending_it is not None, 'Call restart first'   
        assert not self.hasFinished(), 'You should not call step() if the optimization had finished'
        # Each iteration chooses one of the neighbors of current_sol
        neighbors = self.neighbors_fn(self.domains, self.current_sol, self.grid_steps)
        # First try to randomly pick a NOT vetoed candidate: who is not in tl
        unvetoed_neighbors = [candidate for candidate in neighbors if list(candidate) not in self.tl]
        if len(unvetoed_neighbors) > 0:
            next_candidate = random.choice(unvetoed_neighbors)
            next_candidate_cost = self.cost_fn(next_candidate)
        # Otherwise uses the aspiration criteria and chooses the best vetoed neighbor
        else:
            # next_candidate = max(neighbors, key = cost_fn)
            # next_candidate_cost = cost_fn(next_candidate)
            next_candidate, next_candidate_cost = None, np.inf
            for candidate in neighbors:
                candidate_cost = self.cost_fn(candidate)
                if candidate_cost < next_candidate_cost:
                    next_candidate, next_candidate_cost = candidate, candidate_cost
        # Update the best_sol, if a better candidate is found
        if next_candidate_cost < self.best_cost:
            self.best_sol, self.best_cost = next_candidate, next_candidate_cost
            better_sol_found = True
        else:
            better_sol_found = False
        # Anyway, update the current_sol
        self.current_sol = next_candidate
        # Veto the candidate
        self.tl.append(list(next_candidate))
        # Limit the size of the tl
        if len(self.tl) > self.max_tl_len:
            self.tl = self.tl[len(self.tl)//2:]
        if self.pending_it == self.cooling_it:
            self.applyCooling()
        self.pending_it -= 1
        return better_sol_found
        
    def hasFinished(self):
        return self.pending_it<=0

def sphere_cost(sol) -> float:
    """ Return the cost of the shpere problem """
    return sum([v**2.0 for v in sol])

if __name__ == '__main__':
    # Perform the test
    dimensions = 3
    domains = [(-5.0,5.0)] * dimensions
    grid_percentage = 0.05
    grid_steps = grid_percentage * np.array((10, 10, 10))
    start_sol = [np.random.uniform(domain[0],domain[1])for domain in domains]
    sa = SAOptimizer(domains, sphere_cost, grid_steps)
    finished = False
    sa.restart(start_sol, T = 100000.0)
    while (not finished):
        better_sol_found = sa.step()
        if better_sol_found :
            print('SA BETTER SOL T:%.2f E:%.3f Sol:' % (sa.T, sa.best_E), sa.best_sol)
        finished = sa.hasFinished()
    res = (sa.best_E, sa.best_sol)
    print("Sphere sa", res)
    ts = TSOptimizer(domains, sphere_cost, grid_steps)
    finished = False
    ts.restart(start_sol, max_it=100000)
    while (not finished):
        better_sol_found = ts.step()
        if better_sol_found :
            print('TS BETTER SOL pending_it:%d cost:%.3f Sol:' % (ts.pending_it, ts.best_cost), ts.best_sol)
        finished = ts.hasFinished()
    res = (ts.best_cost, ts.best_sol)
    print("Sphere ts", res)
    
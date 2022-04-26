import numpy as np
import random
import math

class Optimizer:
    def __init__(self, domains, grid_steps):
        self.domains = domains
        self.grid_steps = grid_steps
        self.neighbors_fn = Optimizer.grid_neighbors
    @staticmethod
    def grid_neighbors(domains, sol, grid_steps):
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

class SAOptimizer (Optimizer):
    def __init__(self, domains, cost_fn, grid_steps):
        super().__init__(domains, grid_steps)
        self.cost_fn = cost_fn
        self.has_cooled = False
        
    def restart(self, start_sol, T=10000.0, cool_factor = 0.99, stopT = 1.0):
        self.T = T
        self.cool_factor = cool_factor
        self.stopT = stopT
        self.best_sol = self.current_sol = np.array(start_sol)
        self.best_E  = self.cost_fn(start_sol)
        
    def step(self):
        """ Executes one optimization iteration """
        """ Returns If a better solution have been found during this iteration """
        """         In this case the solution has to be draw """
        """         Then best_sol, bestE contain the best solution found """
        assert self.T is not None, 'Call restart first'
        assert self.T >= self.stopT, 'You should not call step() if the optimization had finished'
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
            for i in range(len(self.grid_steps)):
                self.grid_steps[i] = self.grid_steps[i]/5
            self.current_sol = self.best_sol
            self.has_cooled = True
        # Decrease temperature
        self.T = self.cool_factor * self.T
        return better_sol_found

    def hasFinished(self):
        return self.T<self.stopT

def sphere_cost(sol):
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
    sa.restart(start_sol)
    while (not finished):
        better_sol_found = sa.step()
        if better_sol_found :
            print('BETTER SOL T:%.2f E:%.3f Sol:' % (sa.T, sa.best_E), sa.best_sol)
        else:
            print('Current Sol T:%.2f E:%.3f Sol:' % (sa.T, sa.cost_fn(sa.current_sol)), sa.current_sol)
        finished = sa.hasFinished()
    res = (sa.best_E, sa.best_sol)
    print("Sphere sa", res)

    
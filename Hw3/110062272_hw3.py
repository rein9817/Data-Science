import numpy as np
from HomeworkFramework import Function  # you must import this class "Function"

class CMAES_optimizer(Function):  # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func)  # must have this init to work normally
        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)
        self.target_func = target_func
        self.eval_times = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)
        
    def get_optimal(self):
        return self.optimal_solution, self.optimal_value
    
    def run(self, FES):  # main part for your implementation
        lambda_ = 4 + int(3 * np.log(self.dim))  # population size
        mu = lambda_ // 2  # number of parents
        
        # Weight initialization
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)  # normalize weights
        
        # Calculate effective mu
        mu_eff = 1 / np.sum(weights**2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff/self.dim) / (self.dim + 4 + 2*mu_eff/self.dim)
        c1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        # Initialize dynamic strategy parameters
        p_sigma = np.zeros(self.dim)  # evolution path for sigma
        p_c = np.zeros(self.dim)  # evolution path for C
        C = np.eye(self.dim)  # covariance matrix
        
        # Initial mean and step size
        mean = np.random.uniform(self.lower, self.upper, self.dim)
        sigma = 0.3 * (self.upper - self.lower)  # step size
        
        # Eigendecomposition variables
        B = np.eye(self.dim)  # B contains the eigenvectors
        D = np.ones(self.dim)  # D contains the eigenvalues (diagonal)
        BD = B  # BD = B*D to sample, see below
        
        # Number of iterations already done
        g = 0
        
        # Main loop
        while self.eval_times < FES:
            print('=====================FE=====================')
            print(self.eval_times)
            
            # Create new population of search points and evaluate them
            x = np.zeros((lambda_, self.dim))
            y = np.zeros((lambda_, self.dim))
            f_values = np.zeros(lambda_)
            
            for k in range(lambda_):
                # Sample new solution
                z = np.random.normal(0, 1, self.dim)
                y[k] = np.dot(B, D * z)  # y_k = B*D*z_k
                x[k] = mean + sigma * y[k]  # x_k = m + sigma*y_k
                
                # Ensure bounds are respected
                x[k] = np.clip(x[k], self.lower, self.upper)
                
                # Evaluate function
                f_values[k] = self.f.evaluate(self.target_func, x[k])
                self.eval_times += 1
                
                # Update optimal solution if better
                if f_values[k] != "ReachFunctionLimit" and float(f_values[k]) < self.optimal_value:
                    self.optimal_solution[:] = x[k]
                    self.optimal_value = float(f_values[k])
                
                # Check if evaluation limit reached
                if f_values[k] == "ReachFunctionLimit" or self.eval_times >= FES:
                    print("ReachFunctionLimit")
                    return
            
            # Sort by fitness
            sorted_indices = np.argsort(f_values)
            
            # Calculate weighted mean of selected points
            y_w = np.zeros(self.dim)
            for i in range(mu):
                idx = sorted_indices[i]
                y_w += weights[i] * y[idx]
            
            # Update mean
            old_mean = mean.copy()
            mean = mean + sigma * y_w
            
            # Update evolution paths
            h_sigma = 0
            ps_norm = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (g + 1)))
            if ps_norm < (1.4 + 2/(self.dim + 1)) * np.sqrt(self.dim):
                h_sigma = 1
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.dot(B, y_w)
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * np.dot(B, y_w)
            
            # Adapt covariance matrix
            w_io = np.ones(mu)  # Weights for C matrix update
            
            # Rank-mu update
            rank_mu_update = np.zeros((self.dim, self.dim))
            for i in range(mu):
                idx = sorted_indices[i]
                rank_mu_update += weights[i] * np.outer(y[idx], y[idx])
            
            # Full covariance matrix update
            C = (1 + c1 * (1 - h_sigma) - c1 - c_mu * np.sum(weights)) * C + \
                c1 * np.outer(p_c, p_c) + c_mu * rank_mu_update
            
            # Update step size
            sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(self.dim) - 1))
            
            # Update B and D from C (eigendecomposition)
            if g % 1 == 0:  # Perform eigendecomposition every iteration (adjust frequency as needed)
                C = np.triu(C) + np.triu(C, 1).T  # Enforce symmetry
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(1e-10, D))  # D contains the standard deviations
                BD = B * D
            
            g += 1
            print("optimal: {}\n".format(self.get_optimal()[1]))
        
if __name__ == '__main__':
    func_num = 1
    fes = 0
    #function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000
        else:
            fes = 2500
        
        # you should implement your optimizer
        op = CMAES_optimizer(func_num)
        op.run(fes)
        
        best_input, best_value = op.get_optimal()
        print(best_input, best_value)
        
        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1
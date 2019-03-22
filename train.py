import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle
import sys


class PSO_SVR:
    def __init__(self,
                 data_file,
                 n_svrs = 4,
                 kernel = "rbf",
                 n_iterations = 1,
                 validation_size = 0.1,
                 n_validations = 1):
    
        self.n_svrs = n_svrs
        self.n_iterations = n_iterations
        self.inertia_weight = 1
        self.c1 = 2
        self.c2 = 2
        self.n_validations = n_validations
        self.validation_size = validation_size

        # List of all the SVRs (particles).
        self.svrs = []
        # List containing the  best error values of each svr.
        self.p_best_err= []
        # List containing the parameters which gave the best error for each SVR.
        self.p_best_params = []
        # The global best value of error.
        self.g_best_err = np.inf
        # The parameters which gave the global best error value.
        self.g_best_params = {"C":0, "epsilon": 0}

        # Initialize all the particle states and p_best values.
        for i in range(n_svrs):
            C = np.random.uniform(low = 0.1, high = 100)
            epsilon = np.random.uniform(low = 0, high = 1)
            gamma = 'auto'
            svr = SVR(kernel = kernel,
                       C = C,
                       epsilon = epsilon,
                       gamma = gamma)

            self.svrs.append(svr)
            self.p_best_err.append(np.inf)
            self.p_best_params.append({"C":0, "epsilon": 0})

        # Load the data
        self.load_data(data_file)


    def load_data(self, data_file):
        col_names = ['x1', 'x2', 'x3', 'y']
        df = pd.read_csv(data_file, names = col_names)

        self.X = df[['x1', 'x2', 'x3']].values
        self.Y = df[['y']].values


    def run_optimizer(self):
        for i in range(self.n_iterations):
            print("\nOptimizer step: ", i+1)

            # Train the SVRs and obtain the validation errors.
            self.train_svrs()
                
            # Update the state of the particles (C and epsilon) based on the
            # validation errors.
            self.update_particle_state()

            print("Step best error: ", min(self.val_error))
            print("Global best error: ", self.g_best_err)


    def train_svrs(self):
        # List of list to store the validation error of every SVR. Each sublist
        # corresponds to one svr and each element in the sublist will be
        # the validation error corresponding to one validation step.
        val_error_list = [[] for i in range(self.n_svrs)]

        for validation_step in range(self.n_validations):
            # Split the data into train and validation.
            x_train, x_val, y_train, y_val = train_test_split(
                        self.X, self.Y, test_size = self.validation_size)

            # Input normalization.
            # Scale x_train to have 0 mean and 1 SD.
            x_scaler = StandardScaler()
            x_train = x_scaler.fit_transform(x_train)
            # Scale x_val using mean and SD of x_train.
            x_val = x_scaler.transform(x_val)

            for j in range(self.n_svrs):
                # Train the SVR.
                self.svrs[j].fit(x_train, y_train.ravel())

                # Get validation error.
                y_pred = self.svrs[j].predict(x_val)
                val_error = mean_squared_error(y_val, y_pred)

                # Store the error.
                val_error_list[j].append(val_error)

        # Set final validation error as the mean of per step validation errors.
        self.val_error = np.mean(val_error_list, axis = 1) 

    
    def update_particle_state(self):
        # Update the personal best and global best values.
        for i in range(self.n_svrs):
            # Update p_best values if required.
            if self.val_error[i] < self.p_best_err[i]:
                params = self.svrs[i].get_params()
                self.p_best_err[i] = self.val_error[i]
                self.p_best_params[i] = {'C': params['C'],
                            'epsilon': params['epsilon']}

                # Update g_best values if required.
                if self.p_best_err[i] < self.g_best_err:
                    self.g_best_err = self.p_best_err[i]
                    self.g_best_params = self.p_best_params[i]

        # Debug
        # print("Error :", self.val_error)
        # print("p_best_err: ", self.p_best_err)
        # print("p_best_params: ", self.p_best_params)
        # print("g_best_err: ", self.g_best_err)
        # print("g_best_params: ", self.g_best_params)

        # Update the parameters of the SVR.
        for i in range(self.n_svrs):
            params = self.svrs[i].get_params()

            # Random numbers to be used in update rule.
            r1 = np.random.random()
            r2 = np.random.random()

            # Find new C value.
            C = params["C"]
            C_new = self.inertia_weight * C \
                    + r1*self.c1*(self.p_best_params[i]["C"] - C) \
                    + r2*self.c2*(self.g_best_params["C"] - C)
            C_new = max(0.01, C_new)

            # Find new epsilon value.
            epsilon = params["epsilon"]
            epsilon_new = self.inertia_weight * epsilon \
                    + r1*self.c1*(self.p_best_params[i]["epsilon"] - epsilon) \
                    + r2*self.c2*(self.g_best_params["epsilon"] - epsilon)
            epsilon_new= max(0.00001, epsilon_new)

            # Update the parameters in SVR.
            self.svrs[i].set_params(**{'C': C_new, 'epsilon': epsilon_new})


    def get_best_values(self):
        return self.g_best_err, self.g_best_params

#----------------------------PARAMETERS--------------------------------------         
DATA_FILE_PATH = "data/train_data.csv"
N_SVRS = 50
N_ITERATIONS = 100
N_VALIDATIONS = 10
VALIDATION_SIZE = 0.1
SVR_KERNEL = "rbf"
#----------------------------------------------------------------------------

if __name__ == "__main__":
    pso_optimizer = PSO_SVR(data_file = DATA_FILE_PATH,
                      n_svrs = N_SVRS,
                      kernel = SVR_KERNEL,
                      n_iterations = N_ITERATIONS,
                      validation_size = VALIDATION_SIZE,
                      n_validations = N_VALIDATIONS)
    pso_optimizer.run_optimizer()
    best_err, best_params = pso_optimizer.get_best_values()
    print("Best error :", best_err)
    print("Best params:", best_params)


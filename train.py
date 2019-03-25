import configparser
import numpy as np
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from utils import load_data


#---------------------------PSO_SVR class-------------------------------------#

class PSO_SVR:
    """
    Class which trains a set of SVRs on a given dataset and identifies 
    the best hyper parameters C and gamma of the SVR which gives best
    result.
    """

    def __init__(self, config_file):
        # Parse the config file.
        config = configparser.ConfigParser()
        config.read(config_file)

        # PSO parameters.
        pso_config          = config["PSO"]
        self.n_iterations   = int(pso_config.get("n_iterations", 100)) 
        self.inertia_wt     = float(pso_config.get("inertia_weight", 1))
        self.c1             = float(pso_config.get("c1", 2)) 
        self.c2             = float(pso_config.get("c2", 2)) 

        # Validation parameters.
        val_config          = config["Validation"]
        self.validation_type= val_config.get("type", "random-split")
        self.n_validations  = int(val_config.get("n_validations", 10))
        self.validation_size= float(val_config.get("validation_size", 0.1))

        # SVR parameters.
        svr_config          = config["SVR"]
        self.kernel         = svr_config.get("kernel", "rbf")
        self.epsilon        = float(svr_config.get("epsilon", 0.1))
        self.C_min          = float(svr_config.get("C_min", 1))
        self.C_max          = float(svr_config.get("C_max", 100))
        self.C_step         = float(svr_config.get("C_step", 10))
        self.gamma_min      = float(svr_config.get("gamma_min", 0.01))
        self.gamma_max      = float(svr_config.get("gamma_max", 2))
        self.gamma_step     = float(svr_config.get("gamma_step", 0.1))

        # Data file
        data_config         = config["Data"]

        # Load the data
        self.X, self.Y = load_data(data_config)

        # Initialze all the PSO particles (SVRs).
        self.init_particles()
        print("Number of SVRs :", self.n_svrs)



    def run_optimizer(self):
        """ 
        Driver function which starts the PSO optimization.
        """
        for i in range(self.n_iterations):
            print("\nOptimizer step: ", i+1)

            # Train the SVRs and obtain the validation scores.
            self.train_svrs()
                
            # Update the state of the particles (C and gamma) based on the
            # validation scores.
            self.update_particle_state()

            print("Step best score: ", max(self.val_score))
            print("Global best score: ", self.g_best_score)


    def train_svrs(self):
        """
        Funtion to train all the SVRs and obtain their validation scores.
        """
        # List of list to store the validation score of every SVR. Each sublist
        # corresponds to one svr and each element in the sublist will be
        # the validation score corresponding to one validation step.
        val_score_list = [[] for i in range(self.n_svrs)]

        # Train each 
        for val_step in range(self.n_validations):
            # Get the training and validation data.
            x_train, x_val, y_train, y_val = self.get_train_val_data(
                        validation_step = val_step + 1) 

            for j in range(self.n_svrs):
                # Train the SVR.
                self.svrs[j].fit(x_train, y_train.ravel())

                # Get validation score.
                # y_pred = self.svrs[j].predict(x_val)
                # val_score = mean_squared_score(y_val, y_pred)
                val_score = self.svrs[j].score(x_val, y_val)

                # Store the score.
                val_score_list[j].append(val_score)

        # Set final validation score as the mean of per step validation scores.
        self.val_score = np.mean(val_score_list, axis = 1) 

    
    def update_particle_state(self):
        """ 
        Updates the personal and global best values of every particle and then
        updates the parameters of each particle.
        """
        # Update the personal best and global best values.
        for i in range(self.n_svrs):
            # Update p_best values if required.
            if self.val_score[i] > self.p_best_score[i]:
                params = self.svrs[i].get_params()
                self.p_best_score[i] = self.val_score[i]
                self.p_best_params[i] = {'C': params['C'],
                            'gamma': params['gamma']}

                # Update g_best values if required.
                if self.p_best_score[i] > self.g_best_score:
                    self.g_best_score = self.p_best_score[i]
                    self.g_best_params = params


        # Update the parameters of the SVRs.
        for i in range(self.n_svrs):
            params = self.svrs[i].get_params()

            # Random numbers to be used in update rule.
            r1 = np.random.random()
            r2 = np.random.random()

            # Find new C value.
            C = params["C"]
            C_new = self.inertia_wt * C \
                    + r1*self.c1*(self.p_best_params[i]["C"] - C) \
                    + r2*self.c2*(self.g_best_params["C"] - C)
            C_new = max(0.01, C_new)

            # Find new gamma value.
            gamma = params["gamma"]
            gamma_new = self.inertia_wt * gamma \
                    + r1*self.c1*(self.p_best_params[i]["gamma"] - gamma) \
                    + r2*self.c2*(self.g_best_params["gamma"] - gamma)
            gamma_new= max(0.001, gamma_new)

            # Update the parameters in SVR.
            self.svrs[i].set_params(**{'C': C_new, 'gamma': gamma_new})


    def init_particles(self):
        """
        Initializes all the particels, personal best and global best values.
        For SVR initialization, the set of C values: from C_min (inclusive) to
        C_max(exclusive) in steps of C_step, and the set of gamma values: 
        from eps_min(inclusive) to eps_max(exclusive) in steps of eps_step,
        are taken, then an SVR is initialized for every possible (C, gamma)
        pair.
        """ 
        # List of all the SVRs (particles).
        self.svrs = []
        # List containing thebest score values of each svr.
        self.p_best_score = []
        # List containing the parameters which gave the best score for each SVR.
        self.p_best_params = []
        # The global best value of score.
        self.g_best_score = 0
        # The parameters which gave the global best score value.
        self.g_best_params = {}

        # Initialize all the particle states and p_best values.
        for C in np.arange(self.C_min, self.C_max, self.C_step):
            for gamma in np.arange(self.gamma_min, self.gamma_max, self.gamma_step):
                svr = SVR(kernel = self.kernel,
                           C = C,
                           epsilon = self.epsilon,
                           gamma = gamma)

                self.svrs.append(svr)
                self.p_best_score.append(0)
                self.p_best_params.append({"C":0, "gamma": 0})

        self.n_svrs = len(self.svrs)


    def get_train_val_data(self, validation_step):
        """
        Splits self.data into training and validation sets. Also normalizes
        x_train to have 0 mean and 1 standard deviation.

        If validation_type = random-split, randomly splits into 2 with
        validation set size = self.validation_size.

        If validation_type = k-fold, chooses appropriate data subset as 
        validation set (based on parameter validation_step).
        """
        
        if self.validation_type == "random-split":
            x_train, x_val, y_train, y_val = train_test_split(
                        self.X, self.Y, test_size = self.validation_size)
        elif self.validation_type == "k-fold":
            # For k-fold validation type, n_validations represents k.
            k = self.n_validations
            n_samples = self.Y.shape[0]
            subset_size = np.ceil(n_samples/k)

            idx_min = int((validation_step - 1) * subset_size)
            idx_max = validation_step * subset_size
            idx_max = int(min(idx_max, n_samples))

            x_val = self.X[idx_min:idx_max]
            y_val = self.Y[idx_min:idx_max]

            x_train = self.X[np.r_[0:idx_min, idx_max:n_samples]]
            y_train = self.Y[np.r_[0:idx_min, idx_max:n_samples]]

        # Normalize the data.
        # Scale x_train to have 0 mean and 1 SD.
        x_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x_train)
        # Scale x_val using mean and SD of x_train.
        x_val = x_scaler.transform(x_val)

        return x_train, x_val, y_train, y_val


    def get_best_values(self):
        """
        Returns the best score obtained and the parameters of the SVR
        which obtained the best score
        """
        return self.g_best_score, self.g_best_params

#-----------------------------------------------------------------------------#

#--------------------------------Final Trainer--------------------------------#

def train_svr(config_file, svr_params):
    """
    Function to train an SVR with given params using the whole training data
    and save it.
    """
    # Load the data.
    config = configparser.ConfigParser()
    config.read(config_file)
    data_config = config["Data"]
    X, Y = load_data(data_config)

    # Create the whole model pipeline.
    # Scaler to scale X to have 0 mean and 1 standard deviation.
    x_scaler = StandardScaler()
    # SVR model.
    svr = SVR(**svr_params)

    pipeline = Pipeline(steps = [('preprocess', x_scaler), ('SVR', svr)])
    pipeline.fit(X,Y)

    # Save the pipeline.
    save_path = config["Model"]["save_path"]
    with open(save_path, 'wb') as save_file:
        pickle.dump(pipeline, save_file)

#-----------------------------------------------------------------------------#

#-------------------------------------MAIN------------------------------------#

if __name__ == "__main__":

    config_file = sys.argv[1]
    
    # Use PSO to find best hyper parameters for SVR.
    print("Running PSO")
    pso_optimizer = PSO_SVR(config_file = config_file)
    pso_optimizer.run_optimizer()
    best_score, best_params = pso_optimizer.get_best_values()
    print("\nBest score: ", best_score)
    print("Params of best SVR : ", best_params)

    # Use the beset parameters to train an SVR using the whole data.
    print("\nTraining final model")
    train_svr(config_file, best_params)

#-----------------------------------------------------------------------------#

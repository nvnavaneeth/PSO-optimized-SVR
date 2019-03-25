import configparser
import pickle
import numpy as np
import sys
from utils import load_data
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


def test(config_file):
    # Parse config file.
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # Load the data.
    data_config = config["Data"]
    test_x, test_y = load_data(data_config)

    # Load the prediction pipeline.
    model_config = config["Model"]
    with open(model_config["model_path"], "rb") as model_file:
        pipeline = pickle.load(model_file)

    # Calculate test score(coefficient of determination).
    test_score = pipeline.score(test_x, test_y)

    print("Test score(coefficient of determination) :", test_score)



if __name__ == "__main__":
    config_file = sys.argv[1]

    test(config_file)


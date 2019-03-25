import numpy as np

#------------------------Utility Functions------------------------------------#

def get_feauture_cols_list(feature_cols_str):
    """
    Converts the string of comma separated integers to a list of the integers
    """
    ftr_cols = feature_cols_str.split(",")
    ftr_cols = [int(col) for col in ftr_cols]

    return ftr_cols
    

def load_data(data_config):
    """
    Reads data from a csv file and returns X and Y separately.
    """
    data_file = data_config["file_path"]
    feature_cols = get_feauture_cols_list(data_config["feature_cols"])
    label_col = int(data_config["label_col"])

    data = np.genfromtxt(data_file, delimiter = ",")

    return data[:,feature_cols], data[:,label_col]

#-----------------------------------------------------------------------------#


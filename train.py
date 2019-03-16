import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import sys


def load_data(file_path):
    col_names = ['x1', 'x2', 'x3', 'y']
    df = pd.read_csv(file_path, names = col_names)

    return df


class Trainer:
    def __init__(self,
                data_file,
                validation_size = 0.1,
                kernel = 'rbf',
                C = 100,
                gamma = 'auto',
                epsilon = 1,
                degree = 3):

        self.svr = SVR(kernel = kernel,
                       C = C,
                       epsilon = epsilon,
                       gamma = gamma,
                       degree = degree)
        self.validation_size = validation_size

        self.load_data(data_file)
        self.normalize_data()


    def load_data(self, data_file):
        col_names = ['x1', 'x2', 'x3', 'y']
        df = pd.read_csv(data_file, names = col_names)

        # Split into training and validation set.
        train_data, val_data = train_test_split(df, test_size = self.validation_size)

        self.x_train = train_data[['x1', 'x2', 'x3']].values
        self.y_train = train_data[['y']].values
        self.x_val= val_data[['x1', 'x2', 'x3']].values
        self.y_val= val_data[['y']].values


    def normalize_data(self):
        # Scale x_train to have 0 mean and 1 SD.
        self.x_scaler = StandardScaler()
        self.x_train = self.x_scaler.fit_transform(self.x_train)
        # Scale x_val using mean and SD of x_train.
        self.x_val = self.x_scaler.transform(self.x_val)

        # Scale y_train to have 0 mean and 1 SD.
        self.y_scaler = StandardScaler()
        self.y_train = self.y_scaler.fit_transform(self.y_train)
        # Scale y_val using mean and SD of y_train.
        self.y_val = self.y_scaler.transform(self.y_val)

    def train(self):
        pass

if __name__ == "__main__":
    data_file = sys.argv[1]
    trainer = Trainer(data_file)

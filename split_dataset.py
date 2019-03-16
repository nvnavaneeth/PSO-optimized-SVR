import numpy as np
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


TEST_SIZE = 0.1


def split_data(file_path):
    df = pd.read_csv(file_path, header = None)

    train_data, test_data = train_test_split(df, test_size = TEST_SIZE) 

    train_data.to_csv("./data/train_data.csv", header = None, float_format = '%g', index = False)
    test_data.to_csv("./data/test_data.csv", header = None, float_format = '%g', index = False)


if __name__ == '__main__':
    file_path = sys.argv[1]

    split_data(file_path)



import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys


def load_data(file_path):
    col_names = ['x1', 'x2', 'x3', 'y']
    df = pd.read_csv(file_path, names = col_names)

    return df


def main(file_path):
    df = load_data(file_path)

    df.plot(x = 'x1', y = 'y', kind = 'scatter', title = 'x1 vs y')
    df.plot(x = 'x2', y = 'y', kind = 'scatter', title = 'x2 vs y')
    df.plot(x = 'x3', y = 'y', kind = 'scatter', title = 'x3 vs y')

    plt.show()

if __name__ == "__main__":
    file_path = sys.argv[1]

    main(file_path)

"""
Rewrote by Charlie 5/3, making it a dummy class, just to make train_MC.py runnable.
Will simply disable all functionalities of scaler, and make offset 0 and scale 1.
"""

import numpy as np
import pandas as pd
import csv
import time
from logging import raiseExceptions

class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0

        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim, method):
        """
        Args:
            method: 
                'recording': record current run, to get mean and variance for future runs; while using
                    0 and 1 for offset and scale for current run
                'zero_one': 0 and 1 for offset and scale, not recording
                'read_in': use given file to load mean and variance
        """
        self.method = method
        self.obs_dim = obs_dim
        self.scale = None
        self.offset = None

        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True

        if self.method == 'read_in':
            print('Reading in recorded scaler')
            read_in_df = pd.read_csv('./scaler_record/scale_record_20220708-100cars100eps.csv')

            self.offset = read_in_df['means'].values
            epsilon = max(min(read_in_df['stddevs'].min(), .1), 1e-4)
            self.scale = 1. / (read_in_df['stddevs'].values + epsilon) / 3.
            del read_in_df


    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)

        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n


    def get(self):
        """ returns 2-tuple: (scale, offset) """
        if self.method == "zero_one" or self.method == "recording":
            return 0.0, 1.0
        elif self.method == "read_in":
            return self.offset, self.scale
        else:
            raiseExceptions('Cannot recognize this scaling method.')


    def output_to_csv(self):
        """
        In "recording" mode, we want to save the final offset means and stddevs to a csv for future use.
        """
        self.csv_file_addr = './scaler_record/scale_record_' + time.strftime("%Y%m%d-%H%M%S") + '.csv'
        df = pd.DataFrame()
        df['means'] = self.means 
        df['stddevs'] = np.sqrt(self.vars)
        df.to_csv(self.csv_file_addr, index=False)


def main():
    scaler = Scaler('read_in')
    offset, scale = scaler.get()
    print('Offset: \n', offset[80:90])
    print('Scale: \n', scale[80:90])


if __name__ == '__main__':
    main()

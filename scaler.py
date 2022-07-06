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

    def __init__(self, method):
        """
        Args:
            method: 
                'recording': record current run, to get mean and variance for future runs; while using
                    0 and 1 for offset and scale for current run; will use a jupyter script to calculate scale and offset
                'zero_one': 0 and 1 for offset and scale, not recording
                'read_in': use given file to load mean and variance
        """
        self.method = method
        self.scale = None
        self.offset = None

        if self.method == 'recording':
            self.csv_file_addr = './scaler_record/scale_record_' + time.strftime("%Y%m%d-%H%M%S") + '.csv'
            header = np.arange(self.obs_dim)
            with open(self.csv_file_addr, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
        elif self.method == 'read_in':
            read_in_df = pd.read_csv('scaler.csv')
            self.offset = read_in_df['means'].values
            epsilon = max(min(read_in_df['stddevs'].min(), .1), 1e-4)
            self.scale = 1. / (read_in_df['stddevs'].values + epsilon) / 3.
            del read_in_df


    def get(self):
        """ returns 2-tuple: (scale, offset) """
        if self.method == "zero_one" or self.method == "recording":
            return 0.0, 1.0
        elif self.method == "read_in":
            return self.offset, self.scale
        else:
            raiseExceptions('Cannot recognize this scaling method.')


    def build_csv(self, obs):
        """
        Build the csv file that will record the scale and offset for each observation dim and disc_sum_rew
        """
        f = open(self.csv_file_addr, 'ab')
        np.savetxt(f, obs, delimiter=",")
        f.close()



def main():
    scaler = Scaler('read_in')
    offset, scale = scaler.get()
    print('Offset: \n', offset[80:90])
    print('Scale: \n', scale[80:90])


if __name__ == '__main__':
    main()

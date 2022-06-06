import logging
import matplotlib.pyplot as plt
import numpy as np

"""
Created by Charlie 5/3/22, just for a placeholder so that the file can run.
Ask Mark for the actual logger file (or utils).
"""

class Logger(object):
    def __init__(self, logname, now, time_start):
        self.logname = logname
        self.now = now 
        self.time_start = time_start 
        self.path_weights = 'path_weights/'
        self.plot_directory = 'plot_output/'
        self.matching_rates = []
    
    def write(self, display):
        pass

    def close(self):
        pass

    def log(self, input_dict):
        pass

    def log_matching_rate(self, avg_matching_rate, is_last_iteration):
        """
        Input is the avg_matching_rate for a give iteration of PPO.
        If is_last_iteration is true, will plot the 
        """
        self.matching_rates.append(avg_matching_rate)
        if is_last_iteration:
            total_iter = len(self.matching_rates)
            plt.plot(self.matching_rates)
            plt.bar(np.arange(total_iter), self.matching_rates, width=0.1)
            plt.xticks(np.arange(total_iter))
            plt.xlabel('Iteration')
            plt.ylabel('Average Matching Rate over All Agents')
            plt.savefig(self.plot_directory + "matching_rate_over_iterations.png", dpi=200)
            plt.clf()
            

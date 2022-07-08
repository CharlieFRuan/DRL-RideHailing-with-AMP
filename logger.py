import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

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
        self.metric_dict = {'valNNLoss': [], 
                            'policyNNLoss': [],
                            'matching_rates': []}
    
    def write(self, display):
        pass

    def close(self):
        pass

    def log(self, metric_name, metric_val):
        self.metric_dict[metric_name].append(metric_val)

    def log_matching_rate(self, avg_matching_rate, is_last_iteration):
        """
        Input is the avg_matching_rate for a give iteration of PPO.
        If is_last_iteration is true, will plot the 
        """
        self.matching_rates.append(avg_matching_rate)
        if is_last_iteration:
            total_iter = len(self.matching_rates)
            f, ax = plt.subplots(1, figsize=(10, 10))
            plt.bar(np.arange(total_iter) + 1, self.matching_rates)
            plt.xticks(np.insert(np.arange(5, total_iter+1, 5), 0, 1))
            plt.xlabel('Iteration')
            plt.ylabel('Average Matching Rate over All Agents')
            plt.title('Matching Rate over Iterations')
            plt.savefig(self.plot_directory + "matching_rate_over_iterations.png", dpi=200)
            plt.clf()
    
    def output_to_csv(self):
        df = pd.DataFrame()
        df['valNNLoss'] = self.metric_dict['valNNLoss']
        df['policyNNLoss'] = self.metric_dict['policyNNLoss']
        df['matching_rates'] = self.metric_dict['matching_rates']
        df.to_csv('./metrics_log/metrics_' + time.strftime("%Y%m%d-%H%M%S") + '.csv')
            

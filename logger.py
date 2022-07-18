import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
from logging import raiseExceptions

"""
Created by Charlie 5/3/22, just for a placeholder so that the file can run.
Ask Mark for the actual logger file (or utils).
"""

class Logger(object):
    def __init__(self, logname, now, time_start, policy_iter):
        self.logname = logname
        self.now = now 
        self.time_start = time_start 
        self.policy_iter = policy_iter
        self.path_weights = 'path_weights/'
        self.plot_directory = 'plot_output/'
        self.output_folder = os.path.join('stats_output/', time.strftime("%Y%m%d-%H%M%S") + "_" + self.logname)
        os.makedirs(self.output_folder)
        self.matching_rates = []
        self.metric_dict_by_iter = {'valNNLoss': [], 
                            'policyNNLoss': [], 
                            'matching_rates': []} # these metrics only have 1 value per iteration

        # these metrics have more than 1 value per iteration
        # when saving to csv, each column represents 1 iteration
        # for now, only support each iteration has the same number of metrics
        self.other_metrics_dict = {'valNNLoss_full': [], # presents the full loss (loss for each training epoch for each policy iteration)
                            'policyNNLoss_full': []}
        for i in range(self.policy_iter):
            for key, value in self.other_metrics_dict.items():
                value.append([])
    
    def write(self, display):
        pass

    def close(self):
        pass

    def log(self, metric_name, metric_val, cur_policy_iter=None):
        is_by_iter = metric_name in self.metric_dict_by_iter.keys()
        is_other_metric = metric_name in self.other_metrics_dict.keys()
        if is_by_iter and is_other_metric:
            raiseExceptions("Unclear metric name, both dictionaries have this")
        if not(is_by_iter or is_other_metric):
            raiseExceptions("Metric name not found")

        if is_by_iter: 
            self.metric_dict_by_iter[metric_name].append(metric_val)
        else:
            self.other_metrics_dict[metric_name][cur_policy_iter].append(metric_val)
        

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
        df['valNNLoss'] = self.metric_dict_by_iter['valNNLoss']
        df['policyNNLoss'] = self.metric_dict_by_iter['policyNNLoss']
        df['matching_rates'] = self.metric_dict_by_iter['matching_rates']
        df.to_csv(os.path.join(self.output_folder, 'metrics_by_iter.csv'))

        # Not an efficient way, but currently not many values, so it is okay
        # TODO: use csv.writer to make more efficient
        for metric_name, data_lists in self.other_metrics_dict.items():
            df = pd.DataFrame()
            for i in range(self.policy_iter):
                df[i+1] = data_lists[i]
            df.to_csv(os.path.join(self.output_folder, metric_name + '.csv'))

            

"""
Defines the agent class.
Each agent has a corresponding PolicyNN, and the class has functions that determines the action,
trains the policyNN, run an episode with the policy.
"""

from logging import raiseExceptions
import ray
from env import Env
from scaler import Scaler
from policy import NNPolicy
import tensorflow_probability as tfp
import numpy as np
from collections import OrderedDict
import argparse

@ray.remote
class Agent(object):
    
    def __init__(self, network: Env, scaler: Scaler, hid1_mult, hid3_size, sz_voc, embed_dim, model_dir):
        """
        network: transporation network, our environment
        scaler: for standardizing/normalizing values
        """
        self.network = network 
        self.scaler=  scaler
        obs_dim = network.obs_dim
        act_dim = network.R * network.R
        self.policy_model = NNPolicy(obs_dim, act_dim, hid1_mult, hid3_size, sz_voc, embed_dim)
        self.policy_model.built = True # to allow loading weights to a just initialized model
        self.policy_model.load_weights(model_dir)
        
    def sample(self, obs, x_t, stochastic=True):
        """
        obs: state
        x_t: time component
        stochastic: stochastic or deterministic policy
        return: if stochastic, returns pi(a|x), else returns distribution with prob=1 on argmax[pi(a|x)]
        """
        # TODO: implement deterministic, if needed
        if stochastic:
            act_prob_out = self.policy_model(obs, x_t)
            dist = tfp.distributions.Categorical(act_prob_out)
            action = dist.sample().numpy()[0]
            return action
        else:
            raiseExceptions('Have not implemented deterministic yet.')


    def run_episode(self):
        """
        One episode simulation.
        Return: collected trajectories
        """
        offset, scale = 0., 1.
        if self.scaler is not None:
            offset, scale = self.scaler.get()
        observes_unscaled, observes, times, actions, rewards = [], [], [], [], []

        # Reset needed; because the input network may have run for a while, and thus 
        # network.car_init_dist is not None but resetting is needed!
        self.network.reset()

        s_running, dec_epoch = self.network.get_initial_state()  # unscaled

        while dec_epoch < self.network.H:
            num_free_nby_cars = self.network.num_avb_nby_cars(s_running)

            while num_free_nby_cars > 0.5:  # cars' perspective
                s_scaled = (s_running - offset) * scale
                trip_type = self.sample(s_scaled, dec_epoch, stochastic=True)
                while True:  # resample until feasible action
                    orig, dest = divmod(trip_type, self.network.R)
                    eta_w_car = np.where(s_running[self.network.car_dims_cum[orig]:\
                        (self.network.car_dims_cum[orig] + self.network.L + 1)] > 0.5)[0]
                    if len(eta_w_car) <= 0:
                        continue  # no free nearby car associated with this orig region -> resample

                    slot_id = self.network.min_to_slot(dec_epoch + eta_w_car[0])
                    # Collect state:
                    observes_unscaled.append(s_running.copy())  # because s_running will change
                    observes.append(s_scaled.copy())
                    # Collect state - time component:
                    times.append(dec_epoch)
                    # Collect action
                    actions.append(trip_type)  # matching or routing can be inferred from the reward
                    if s_running[self.network.car_dim + trip_type - self.network.num_imp_ride[trip_type]] > 0.5:
                        # Car-Passenger matching
                        rewards.append(self.network.c[slot_id][orig, dest])
                        # post-decision state
                        s_running[self.network.car_dim+trip_type-self.network.num_imp_ride[trip_type]] -= 1
                        s_running[self.network.car_dims_cum[dest] + eta_w_car[0] + self.network.tau[slot_id][orig, dest]] += 1
                    elif eta_w_car[0] <= 0.5 and dest != orig:
                        s_running[self.network.car_dims_cum[dest] + eta_w_car[0] + self.network.tau[slot_id][orig, dest]] += 1
                        rewards.append(self.network.tilde_c[slot_id][orig, dest])
                    else:
                        s_running[self.network.car_dims_cum[-1] + dest * (1 + self.network.L) + eta_w_car[0]] += 1
                        rewards.append(0.)  # Assuming network.tilde_c[:][o, o] == 0.

                    # post-decision state
                    s_running[self.network.car_dims_cum[orig]+eta_w_car[0]] -= 1  # One fewer available nearby car
                    num_free_nby_cars -= 1  # processed one free nearby car
                    
                    break # s_running, dec_epoch is the post-decision state now
            
            dec_epoch = self.network.get_next_state(s_running, dec_epoch)  # Passengers unattended leave!
            assert dec_epoch >= self.network.H or s_running[:self.network.car_dim].sum() == self.network.N, 'Car number problematic!'

        matching_rate = -1
        if len(self.network.queue) != 0:
            matching_rate = float(sum(rewards)) / len(self.network.queue) * 100

        trajectory = OrderedDict([('state', observes_unscaled), ('state_scaled', observes), \
            ('state_time', times), ('action', actions), ('reward', rewards), \
            ('number_passengers', len(self.network.queue)), ('matching_rate', matching_rate)])
        
        print('Car-passenger matching rate: {:.2f}%...'.format(trajectory['matching_rate']))

        return trajectory



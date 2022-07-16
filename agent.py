"""
Defines the agent class.
Each agent has a corresponding PolicyNN, and the class has functions that determines the action,
trains the policyNN, run an episode with the policy.
"""

from logging import raiseExceptions
import ray
# from env import Env
from scaler import Scaler
from policy import NNPolicy
from value_function import NNValueFunction
import tensorflow_probability as tfp
import numpy as np
from collections import OrderedDict
import argparse
import tensorflow as tf

@ray.remote
class Agent(object):
    
    def __init__(self, network, policy_weights, valueNN_weights, scaler: Scaler, hid1_mult, \
                hid3_size, sz_voc, embed_dim, model_dir, cur_iter, valNN_train_epoch, use_AMP):
        """
        network: transporation network, our environment
        scaler: for standardizing/normalizing values
        """
        self.network = network 
        self.scaler =  scaler
        obs_dim = network.obs_dim
        self.act_dim = network.R * network.R
        self.policy_model = NNPolicy(obs_dim, self.act_dim, hid1_mult, hid3_size, sz_voc, embed_dim)
        self.valueNN_model = NNValueFunction(obs_dim, hid1_mult, hid3_size, sz_voc, embed_dim, valNN_train_epoch)
        self.use_AMP = use_AMP

        # dummy call to enable setting weights
        if cur_iter == 1:
            self.policy_model.set_weights(policy_weights)
            self.valueNN_model.set_weights(valueNN_weights)
        else: 
            obs, x_t = self.network.get_initial_state()
            obs = obs * 1.0
            self.policy_model(tf.expand_dims(obs,0), tf.expand_dims(x_t,0)) 
            self.policy_model.set_weights(policy_weights)
            self.valueNN_model(tf.expand_dims(obs,0), tf.expand_dims(x_t,0))
            self.valueNN_model.set_weights(valueNN_weights)
        

    def sample_distr(self, obs, x_t, stochastic=True):
        """
        Return a distribution instead of an action because we may need to resample in run_episode
        obs: state
        x_t: time component
        stochastic: stochastic or deterministic policy
        return: if stochastic, returns pi(a|x), else returns distribution with prob=1 on argmax[pi(a|x)]
        """
        # TODO: implement deterministic, if needed
        if stochastic:
            # expand dims because we only have 1 batch here; make shape (a,) into (1,a)
            act_prob_out = self.policy_model(tf.expand_dims(obs,0), tf.expand_dims(x_t,0))
            dist = tfp.distributions.Categorical(act_prob_out)
            return dist, act_prob_out[0].numpy() # [0] because otherwise it is (1,R*R) shape tensor
        else:
            raiseExceptions('Have not implemented deterministic yet.')


    def run_episode(self):
        """
        One episode simulation.
        Return: collected trajectories
        """
        offset, scale = 0.0, 1.0
        if self.scaler is not None:
            offset, scale = self.scaler.get()
        # next_state_expec_vals is for AMP use; calculated here to help efficiency
        observes_unscaled, observes, times, actions, rewards, next_state_expec_vals = [], [], [], [], [], []

        # Reset needed; because the input network may have run for a while, and thus 
        # network.car_init_dist is not None but resetting is needed!
        self.network.reset() # Generate arrival events, car initial positions

        # s_running is the s_cp in env.py, the state, combining s_c, s_p, s_l; have special indexing
        # dec_epoch is simply the current minute (hence epoch)
        s_running, dec_epoch = self.network.get_initial_state()  # unscaled; 

        while dec_epoch < self.network.H:
            # total number of cars within L minutes away from their destination
            num_free_nby_cars = self.network.num_avb_nby_cars(s_running)

            while num_free_nby_cars > 0.5:  # cars' perspective
                # Generate an action (trip) for each available car (I_t), since that is how SDM works
                # Basically use current state (scaled) to get an action from Policy NN, then interpretate
                # the action using our deterministic dynamic (whether it is car-passenger matching, or
                # empty car routing, do nothing, etc.)
                s_scaled = (s_running - offset) * scale
                trip_distr, act_prob_out = self.sample_distr(s_scaled, dec_epoch, stochastic=True)
                while True:  # resample until feasible action
                    trip_type = trip_distr.sample().numpy()[0]
                    orig, dest = divmod(trip_type, self.network.R) # since trip_type = orig * R + dest

                    # 1. Determine if there is a car that can be assocaited with the generated action
                    # eta_w_car is the indices for s_c(reg) where there is a free car within L minutes away from reg
                    # We will assign car eta_w_car[0] as we see in below codes because we simply take the nearest car
                    eta_w_car = np.where(s_running[self.network.car_dims_cum[orig]:\
                        (self.network.car_dims_cum[orig] + self.network.L + 1)] > 0.5)[0]
                    if len(eta_w_car) <= 0:
                        continue  # no free nearby car associated with this orig region -> resample
                    
                    # 2. Collect data
                    # Collect state:
                    observes_unscaled.append(s_running.copy())  # copy because s_running will change
                    observes.append(s_scaled.copy())
                    # Collect state - time component:
                    times.append(dec_epoch)
                    # Collect action
                    actions.append(trip_type)  # matching or routing can be inferred from the reward

                    # 3. Calculate AMP portion (basically do the exact same things in rest of the function, but for all actions)
                    if self.use_AMP:
                        # cur_next_state_expec_val = 0 # cumulator for next state's value function expecation
                        next_state = s_running.copy() # next state is based on current state
                        if num_free_nby_cars == 1:
                            # last car in an SDM, needs special treatment, postpone to train_MC.py
                            next_state_expec_vals.append(None)
                        else: 
                            action_probs = np.zeros(self.act_dim) # the probability for each action
                            action_vals = np.zeros(self.act_dim) # zeta(s') given we take action a
                            for trip_type_i, prob in enumerate(act_prob_out):
                                o, d = divmod(trip_type_i, self.network.R)
                                # 3.1 determine if this action is feasible (i.e. has available car to be associated)
                                eta_w_car_i = np.where(s_running[self.network.car_dims_cum[o]:\
                                            (self.network.car_dims_cum[o] + self.network.L + 1)] > 0.5)[0]
                                if len(eta_w_car_i) <= 0:
                                    # not a feasible action, view it as 0 probability of being picked
                                    continue 
                                action_probs[trip_type_i] = prob # so infeasible ones stay as zero
                                # 3.2 get next state based on how we define system's dynamic
                                slot_id_i = self.network.min_to_slot(dec_epoch + eta_w_car_i[0]) 
                                if next_state[self.network.car_dim + trip_type_i - self.network.num_imp_ride[trip_type_i]] > 0.5:
                                    # Car-Passenger matching, since there is at least 1 passenger requesting such type of trip
                                    # First remove passenger matched from count
                                    next_state[self.network.car_dim+trip_type_i-self.network.num_imp_ride[trip_type_i]] -= 1 
                                    # Add a car that is traveling to dest, eta_w_car[0] is remaining time to finish current trip, tau is traveling time to dest 
                                    next_state[self.network.car_dims_cum[d] + eta_w_car_i[0] + self.network.tau[slot_id_i][o, d]] += 1
                                elif eta_w_car_i[0] <= 0.5 and d != o:
                                    # Empty-Car Routing: no passenger requesting such a trip, but o != g
                                    next_state[self.network.car_dims_cum[d] + eta_w_car_i[0] + self.network.tau[slot_id_i][o, d]] += 1
                                else:
                                    # Do-Nothing
                                    next_state[self.network.car_dims_cum[-1] + d * (1 + self.network.L) + eta_w_car_i[0]] += 1   
                                next_state[self.network.car_dims_cum[o]+eta_w_car_i[0]] -= 1  # One fewer available nearby car

                                # 3.3 now we have s', feed it to value function to get the zeta(s')
                                # not dec_epoch+1 because not last car in SDM
                                next_state = (next_state - offset) * scale
                                # cur_next_state_expec_val += prob * self.valueNN_model([next_state], np.array([dec_epoch]))
                                action_vals[trip_type_i] = self.valueNN_model([next_state], np.array([dec_epoch]))

                            # 3.4 After recording each feasible state's values, calculate the expected zeta value for the next state
                            # Some actions are infeasible even though its action_prob > 0, reweight so that feasible distribution adds up to 1
                            action_probs = action_probs / np.sum(action_probs) # now np.sum(action_probs) should be 1
                            cur_next_state_expec_val = np.dot(action_probs, action_vals)
                            next_state_expec_vals.append(cur_next_state_expec_val)

                    # 4. Now interpretate action based on how we define system's dynamic
                    # determine which time slot it is when the car actually arrives (hence + eta_w_car[0])
                    slot_id = self.network.min_to_slot(dec_epoch + eta_w_car[0]) 
                    if s_running[self.network.car_dim + trip_type - self.network.num_imp_ride[trip_type]] > 0.5:
                        # Car-Passenger matching, since there is at least 1 passenger requesting such type of trip
                        rewards.append(self.network.c[slot_id][orig, dest])
                        # Post-Decision state
                        # First remove passenger matched from count
                        s_running[self.network.car_dim+trip_type-self.network.num_imp_ride[trip_type]] -= 1 
                        # Add a car that is traveling to dest, eta_w_car[0] is remaining time to finish current trip, tau is traveling time to dest 
                        # Note that we will remove this car from orig's s_c after this if-else clause
                        s_running[self.network.car_dims_cum[dest] + eta_w_car[0] + self.network.tau[slot_id][orig, dest]] += 1
                    elif eta_w_car[0] <= 0.5 and dest != orig:
                        # Empty-Car Routing
                        # There is no passenger requesting such a trip; but the available car is 
                        # already idling at orig, and orig and dest are differnet regions
                        s_running[self.network.car_dims_cum[dest] + eta_w_car[0] + self.network.tau[slot_id][orig, dest]] += 1
                        rewards.append(self.network.tilde_c[slot_id][orig, dest])
                    else:
                        # Do-Nothing
                        # There is no passenger requesting such a trip; either the car is not yet 
                        # idling (eta_w_car[0] >= 1), or dest == orig.
                        # Here we edit the s_l part of s_cp according to paper, to exclude it from the available car pool
                        s_running[self.network.car_dims_cum[-1] + dest * (1 + self.network.L) + eta_w_car[0]] += 1
                        rewards.append(0.)  # Assuming network.tilde_c[:][o, o] == 0.

                    # post-decision state
                    # We remove the car from its original destination's s_c, since there is no point in keeping track of that
                    # This also gurantees that s_c's sum is always the total number of cars
                    s_running[self.network.car_dims_cum[orig]+eta_w_car[0]] -= 1  # One fewer available nearby car
                    num_free_nby_cars -= 1  # processed one free nearby car
                    
                    break # s_running, dec_epoch is the post-decision state now
            dec_epoch = self.network.get_next_state(s_running, dec_epoch)  # Passengers unattended leave!
            assert dec_epoch >= self.network.H or s_running[:self.network.car_dim].sum() == self.network.N, 'Car number problematic!'

        matching_rate = -1
        if len(self.network.queue) != 0:
            matching_rate = sum(rewards) * 100.0 / len(self.network.queue) 
            
        trajectory = OrderedDict([('state', observes_unscaled), ('state_scaled', observes), \
            ('state_time', np.array(times)), ('action', actions), ('reward', rewards), \
            ('number_passengers', len(self.network.queue)), ('matching_rate', matching_rate), \
            ('next_state_expec_vals', np.array(next_state_expec_vals)) ])

        print('Car-passenger matching rate: {:.2f}%...'.format(trajectory['matching_rate']))

        return trajectory

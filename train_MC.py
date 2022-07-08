# ML-related libraries
import ray
import tensorflow as tf
import numpy as np

# local modules
from env import Env
from logger import Logger
from scaler import Scaler
from value_function import NNValueFunction, fit_valueNN
from policy import NNPolicy, updatePolicy
from agent import Agent

# python libraries
import argparse
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress TF's log

MAX_ACTORS = 4 # max number of parallel simulations


def run_policy(network_id, policy: NNPolicy, val_func: NNValueFunction, scaler, logger, gamma, cur_iter, episodes, \
    policy_temp_save_dir, valNN_train_epoch, use_AMP):
    """
    Run given policy and collect data
    network_id: queuing network structure and first-order info
    policy: queuing network policy
    scaler: normalization values
    logger: metadata accumulator
    gamma: discount factor (used when normalizing states)
    cur_iter: current policy iteration for the outermost loop in main; used to save weights
    episodes: number of parallel simulations (episodes)
    return: trajectories = (states, actions, rewards)
    """
    # 1. Prepare for parallel actors generation
    # save weights of current policy NN; loaded in agent initialization
    scaler_id = ray.put(scaler)
    # policy.save_weights(policy_temp_save_dir) 
    policy_weight_id = ray.put(policy.get_weights())
    valueNN_weight_id = ray.put(val_func.get_weights())

    simulators = [Agent.remote(network=ray.get(network_id), policy_weights=ray.get(policy_weight_id), \
        valueNN_weights=ray.get(valueNN_weight_id), scaler=ray.get(scaler_id), hid1_mult=1, hid3_size=5, \
        sz_voc=ray.get(network_id).H, embed_dim=ray.get(network_id).num_slots*2, \
        model_dir=policy_temp_save_dir, cur_iter=cur_iter, valNN_train_epoch=valNN_train_epoch, \
        use_AMP=use_AMP) for _ in range(MAX_ACTORS)] 
    run_iterations = episodes // MAX_ACTORS # do not run more parallel processes than number of cores
    remainder = episodes - run_iterations * MAX_ACTORS

    # 2. Start simulation
    start_time = datetime.datetime.now()
    trajectories = []  # list of trajectories
    for j in range(run_iterations):
        trajectories.extend(ray.get([simulators[i].run_episode.remote() for i in range(MAX_ACTORS)]))
    if remainder>0:
        trajectories.extend(ray.get([simulators[i].run_episode.remote() for i in range(remainder)]))

    end_time = datetime.datetime.now()
    sim_time = (end_time - start_time).total_seconds() / 60.0
    print('simulation took: %.2f mins' %(sim_time))

    # 3. Post-process simulation result
    average_reward = np.mean(np.concatenate([t['reward'] for t in trajectories]))
    print('Average cost: ',  -average_reward)

    return trajectories


def discount(x, gamma, v_last):
    """ Calculate discounted forward sum of a sequence at each point """
    disc_array = np.zeros((len(x), 1))
    disc_array[-1] = v_last
    for i in range(len(x) - 2, -1, -1):
        if x[i+1]!=0:
            disc_array[i] = x[i] + gamma * disc_array[i + 1]

    return disc_array


def relarive_af(unscaled_obs, td_pi, lam):
    """
    Calculating one-replication estimate of value function (step 3.5 in paper)
    unscaled_obs: unscaled states for one episode
    td_pi: reward
    lam: 1 here
    """
    # return advantage function
    disc_array = np.copy(td_pi)
    sum_tds = 0
    for i in range(len(td_pi) - 2, -1, -1):
        # Below if else commented out b/c it is doing check that I do not understand; should
        # never enter the else
        # if np.sum(unscaled_obs[i+1]) != 0:
        #     # keeps adding up sum_tds from the back, weight previous totals by lambda
        #     sum_tds = td_pi[i+1] + lam * sum_tds
        # else:
        #     # not sure the point of this if else; how would it be 0 if there are > 0 cars in total?
        #     print("TAKE A LOOK IF THIS LINE GETS PRINTED")
        #     sum_tds = 0

        sum_tds = td_pi[i+1] + lam * sum_tds
        disc_array[i] += sum_tds

    return disc_array

def relarive_af_AMP(rewards, zeta_vals, next_state_expec_vals, lam=1):
    """
    Calculating one-replication estimate of value function (step 3.5 in paper)
    Along with other extra terms in an AMP estimator
    td_pi: reward
    lam: 1 here
    """
    if lam != 1:
        raise(NotImplementedError)
    array_AMP = rewards + next_state_expec_vals - zeta_vals # the terms inside the summation in AMP equation
    cumsum_AMP = np.flip(np.cumsum(np.flip(array_AMP))) + zeta_vals # add current sate's zeta according to equation
    return cumsum_AMP


def add_disc_sum_rew_AMP(trajectories, scaler, cur_iter, H):
    """
    Compute value function for further training of Value Neural Network, use AMP estimator to reduce variance
    trajectories: simulated data
    scaler: normalization values
    cur_iter: current policy iteration for the outermost loop in main
    Case 1: when the state is in the middle of an SDM
    Case 2: when the state is the last car of an SDM, also the last epoch of a trajectory
    Case 3: when the state is the last car of an SDM, but not the last epoch of a trajectory
    """
    # 1. Calculate estimated value for each state
    start_time = datetime.datetime.now()
    for trajectory in trajectories:
        next_state_expec_vals = trajectory['next_state_expec_vals'] # all caclulated except case 3
        state_times = trajectory['state_time']
        rewards = np.array(trajectory['reward'])
        zeta_vals = np.squeeze(trajectory['values'].numpy())
        no_case_2 = False

        if state_times[-1] != H-1:
            # if no state recorded in last epoch, then there is no case 2
            no_case_2 = True

        # 1.3 Finish calculating expectation of next state's value for last cars in an SDM (case 3)
        last_car_ids = None
        if not no_case_2:
            next_state_expec_vals[-1] = 0 # if there is case 2, treat as 0
        last_car_ids = np.where(next_state_expec_vals==None)[0]
        for id in last_car_ids:
            # TODO: simply use zeta(s_{t+1}) to approximate for now
            next_state_expec_vals[id] = zeta_vals[id+1]
        # use astype, otherwise the dtype is object due to None initially
        next_state_expec_vals = next_state_expec_vals.astype('float64') 
        trajectory['disc_sum_rew'] = relarive_af_AMP(rewards, zeta_vals, next_state_expec_vals) 

    end_time = datetime.datetime.now()

    time_took = (end_time - start_time).total_seconds() / 60.0
    print('add_disc_sum_rew took: %.2f mins' %(time_took))

    # 2. Postprocess
    # Burn the last car of each entire episode, also combine all episodes' data into one giant array
    burn = 1 # we do not want the last state and reward for the episode (the entire day)

    disc_sum_rew = np.concatenate([t['disc_sum_rew'][:-burn] for t in trajectories])
    state_times = np.concatenate([t['state_time'][:-burn] for t in trajectories]) 
    observes = np.concatenate([t['state_scaled'][:-burn] for t in trajectories])

    # TODO: scale disc_sum_rew
    # return observes, disc_sum_rew_norm, state_times 

    return observes, disc_sum_rew, state_times 


def add_disc_sum_rew(trajectories, gamma, scaler, cur_iter):
    """
    Compute value function for further training of Value Neural Network
    trajectories: simulated data
    gamma: discount factor
    scaler: normalization values
    cur_iter: current policy iteration for the outermost loop in main
    """
    # 1. Calculate estimated value for each state
    start_time = datetime.datetime.now()
    for trajectory in trajectories:
        if gamma < 1:
            disc_sum_rew = discount(x=trajectory['rewards'],   gamma= gamma, v_last = trajectory['rewards'][-1])
        else:
            disc_sum_rew = relarive_af(trajectory['state'], td_pi=trajectory['reward'], lam=1) 
        trajectory['disc_sum_rew'] = disc_sum_rew
    end_time = datetime.datetime.now()

    time_took = (end_time - start_time).total_seconds() / 60.0
    print('add_disc_sum_rew took: %.2f mins' %(time_took))

    # 2. Postprocess
    # Burn the last car of each entire episode, also combine all episodes' data into one giant array
    burn = 1 # we do not want the last state and reward for the episode (the entire day)
    # unscaled_obs = np.concatenate([t['state'][:-burn] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'][:-burn] for t in trajectories])
    state_times = np.concatenate([t['state_time'][:-burn] for t in trajectories]) 
    observes = np.concatenate([t['state_scaled'][:-burn] for t in trajectories])
    if scaler.method == 'recording':
        scaler.update(observes)

    # TODO: scale disc_sum_rew
    # return observes, disc_sum_rew_norm, state_times 

    return observes, disc_sum_rew, state_times 


def add_value(trajectories, val_func: NNValueFunction, scaler):
    """
    compute value function from the Value Neural Network
    :param trajectory_whole: simulated data
    :param val_func: Value Neural Network
    :param scaler: normalization values
    """
    start_time = datetime.datetime.now()
    # offset, scale = scaler.get()
    for trajectory in trajectories:
        values = val_func(trajectory['state_scaled'], trajectory['state_time'])
        trajectory['values'] = values 
        # trajectory['values'] = values / scale + offset # TODO: scale values? Same as disc_sum_rew
    end_time = datetime.datetime.now()
    time_took = (end_time - start_time).total_seconds() / 60.0
    print('add_value took: %.2f mins' %(time_took))


def build_train_set(trajectories, gamma, scaler):
    """
    Data pre-processing for training
    trajectory_whole:  simulated data
    scaler: normalization values
    return: data for further Policy and Value neural networks training; also calculate average 
    matching rate, since we are already iterating through all the trajectories
    """
    valid_matching_rates = [] # does this to not include rates that are -1 (no ride at all)
    
    # 1. Calculate advantages
    for trajectory in trajectories:
        values = np.squeeze(trajectory['values'])
        unscaled_obs = trajectory['state'] 
        # advantages is calculated using paper's equation 3.7, we append an extra 
        # values[-1] to make size match; last state-action pair will be dropped anyway
        advantages = trajectory['reward'] - values + gamma * np.append(values[1:], values[-1])
        trajectory['advantages'] = np.asarray(advantages)
        rate = trajectory['matching_rate']
        if rate > 0:
            valid_matching_rates.append(rate)

    start_time = datetime.datetime.now()
    burn = 1 # do not want last state in trajectory (last car of the entire day)

    # unscaled_obs = np.concatenate([t['state'][:-burn] for t in trajectories])
    observes = np.concatenate([t['state_scaled'][:-burn] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'][:-burn] for t in trajectories])

    # offset, scale = scaler.get()
    actions = np.concatenate([t['action'][:-burn] for t in trajectories])
    advantages = np.concatenate([t['advantages'][:-burn] for t in trajectories])
    # observes = (unscaled_obs - offset) * scale 
    advantages = advantages  / (advantages.std() + 1e-6) # normalize advantages

    # Now caculate average matching rate
    avg_matching_rate = 0 # 0 if there is no valid rate at all
    if len(valid_matching_rates) != 0:
        avg_matching_rate = np.mean(valid_matching_rates)

    end_time = datetime.datetime.now()
    time_took = (end_time - start_time).total_seconds() / 60.0
    print('build_train_set took: %.2f mins' %(time_took))

    return observes, actions, advantages, disc_sum_rew, avg_matching_rate


def log_batch_stats(trajectories, observes, actions, advantages, disc_sum_rew, logger: Logger, episode, avg_matching_rate, is_last_iteration):

    time_total = datetime.datetime.now() - logger.time_start
    # logger.log({'_mean_act': np.mean(actions),
    #             '_mean_adv': np.mean(advantages),
    #             '_min_adv': np.min(advantages),
    #             '_max_adv': np.max(advantages),
    #             '_std_adv': np.var(advantages),
    #             '_mean_discrew': np.mean(disc_sum_rew),
    #             '_min_discrew': np.min(disc_sum_rew),
    #             '_max_discrew': np.max(disc_sum_rew),
    #             '_std_discrew': np.var(disc_sum_rew),
    #             '_Episode': episode,
    #             '_time_from_beginning_in_minutes': int((time_total.total_seconds() / 60) * 100) / 100.
    #             })
    logger.log_matching_rate(avg_matching_rate, is_last_iteration)
    logger.log('matching_rates', avg_matching_rate)


def main(network_id, num_policy_iterations, gamma, lam, kl_targ, batch_size, hid1_mult, hid3_size,
         clipping_parameter, valNN_train_epoch, policyNN_train_epoch, policy_temp_save_dir, use_AMP, scale_method):
    """
    # Main training loop
    :param: see ArgumentParser below
    """
    now = datetime.datetime.utcnow().strftime("%b-%d_%H-%M-%S")  # create unique directories
    time_start= datetime.datetime.now()
    logger = Logger(logname=ray.get(network_id).network_name, now=now, time_start=time_start)

    scaler = Scaler(ray.get(network_id).obs_dim, scale_method) 
    # Value Neural Network initialization
    val_func = NNValueFunction(obs_dim=ray.get(network_id).obs_dim, hid1_mult=hid1_mult, hid3_size=hid3_size, \
        sz_voc=ray.get(network_id).H, embed_dim=ray.get(network_id).num_slots*2, train_epoch=valNN_train_epoch) 
    # The main Policy Neural Network
    policy = NNPolicy(obs_dim=ray.get(network_id).obs_dim, act_dim=ray.get(network_id).R * ray.get(network_id).R,\
        hid1_mult=hid1_mult, hid3_size=hid3_size, sz_voc=ray.get(network_id).H, embed_dim=ray.get(network_id).num_slots*2, \
        train_epoch=policyNN_train_epoch, reg_str=5e-3, temp=2.0)
    
    iteration = 0  # count of policy iterations
    weights_set = []
    scaler_set = []

    while iteration < num_policy_iterations:
        start_time = datetime.datetime.now()
        # decrease clipping_range and learning rate
        iteration += 1
        alpha = 1. - iteration / num_policy_iterations
        policy.clipping_range = max(0.01, alpha*clipping_parameter)
        policy.lr_multiplier = max(0.05, alpha)

        # Use parallel agent to rollout episodes
        trajectories = run_policy(network_id, policy, val_func, scaler, logger, gamma, iteration, batch_size, policy_temp_save_dir, valNN_train_epoch, use_AMP)
        
        # Calculate value function for each state in each trajectory (step 3.5 in paper; i.e. calculate one-rep estimation of V for each state)
        if use_AMP:
            add_value(trajectories, val_func, scaler) # first use ValNN from i-1 to calculate value, prepare for AMP
            observes, disc_sum_rew_norm, state_times = add_disc_sum_rew_AMP(trajectories, scaler, iteration, ray.get(network_id).H)
        else:
            observes, disc_sum_rew_norm, state_times = add_disc_sum_rew(trajectories, gamma, scaler, iteration)

        # Update value NN with new data
        fit_valueNN(val_func, observes, state_times, disc_sum_rew_norm, logger)
        # Add estimated values to each state in each episode using the updated value NN
        add_value(trajectories, val_func, scaler)  
        # Prepare input for updating policy NN, calculate advantage for each state-action pair
        observes, actions, advantages, disc_sum_rew, avg_matching_rate = build_train_set(trajectories, gamma, scaler)
        # Log statistics
        log_batch_stats(trajectories, observes, actions, advantages, disc_sum_rew, logger, iteration, avg_matching_rate, iteration==num_policy_iterations)
        # Update Policy NN
        updatePolicy(policy, observes, state_times, actions, np.squeeze(advantages), logger)

        logger.write(display=True)  # write logger results to file and stdout
        end_time = datetime.datetime.now()
        time_took = (end_time - start_time).total_seconds() / 60.0
        print('Finished iteration {}/{}'.format(iteration, num_policy_iterations))
        print('This iteration took %.2f minutes' %(time_took))
    
    if scaler.method == 'recording':
        scaler.output_to_csv()
    logger.output_to_csv()

        # TODO: save log, weights, and models


if __name__ == "__main__":
    network = Env()
    network_id = ray.put(network)
    tf.random.set_seed(1)
    np.random.seed(1)

    parser = argparse.ArgumentParser(description=('Train policy for a transportation network '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('-n', '--num_policy_iterations', type=int, help='Number of policy iterations to run',
                        default = 5)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor',
                        default = 1)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default = 1)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default = 0.003)
    parser.add_argument('-b', '--batch_size', type=int, help='Number of episodes per training batch',
                        # default = 300)
                        default = 5)
    parser.add_argument('-m', '--hid1_mult', type=int, help='Multiplier for size of first hidden layer for value and policy NNs',
                        default = 1)
    parser.add_argument('--hid3_size', type=int, help='Size of third hidden layer for value and policy NNs',
                        default = 5)
    parser.add_argument('-c', '--clipping_parameter', type=float, help='Initial clipping parameter',
                        default = 0.2)
    # parser.add_argument('-s', '--skipping_steps', type=int, help='Number of steps for which control is fixed',
    #                     default = 1)
    parser.add_argument('--valNN_train_epoch', type=int, help='Number of epochs to train Value NN',
                        default = 5)
    parser.add_argument('--policyNN_train_epoch', type=int, help='Number of epochs to train Policy NN',
                        default = 3)
    parser.add_argument('--policy_temp_save_dir', type=str, help='Directory to save and load policy NN each iteration',
                        default = './policyNN_temp_save/policy_temp.h5')
    parser.add_argument('--use_AMP', action='store_true', help='flag to run using AMP estimator')
    parser.add_argument('--scale_method', type=str, help='Method of getting scaler',
                        default = 'zero_one')
    args = parser.parse_args()

    print('Starting')
    # main(network_id,  args)
    main(network_id,  **vars(args))
# ML-related libraries
import ray
import tensorflow as tf

# local modules
from env import Env
from logger import Logger
from scaler import Scaler
from value_function import NNValueFunction

# python libraries
import argparse
import datetime



def main(network_id, num_policy_iterations, gamma, lam, kl_targ, batch_size, hid1_mult, episode_duration,
         clipping_parameter, skipping_steps, valNN_train_epoch):
    """
    # Main training loop
    :param: see ArgumentParser below
    """
    now = datetime.datetime.utcnow().strftime("%b-%d_%H-%M-%S")  # create unique directories
    time_start= datetime.datetime.now()
    logger = Logger(logname=ray.get(network_id).network_name, now=now, time_start=time_start)

    scaler = Scaler() # TODO: scaler is a dummy for now
    val_func = NNValueFunction(obs_dim=ray.get(network_id).obs_dim, hid1_mult=1, hid3_size=5, \
        sz_voc=360, embed_dim=6, train_epoch=valNN_train_epoch) # Value Neural Network initialization



if __name__ == "__main__":
    network = Env()

    network_id = ray.put(network)
    tf.random.set_seed(1)


    parser = argparse.ArgumentParser(description=('Train policy for a transportation network '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('-n', '--num_policy_iterations', type=int, help='Number of policy iterations to run',
                        default = 200)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor',
                        default = 1)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default = 1)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default = 0.003)
    parser.add_argument('-b', '--batch_size', type=int, help='Number of episodes per training batch',
                        default = 50) # K=300 in paper? (Charlie 5/3/22)
    parser.add_argument('-m', '--hid1_mult', type=int, help='Size of first hidden layer for value and policy NNs',
                        default = 1)
    parser.add_argument('-t', '--episode_duration', type=int, help='Number of time-steps per an episode',
                        # default = 360)
                        default = 60)
    parser.add_argument('-c', '--clipping_parameter', type=float, help='Initial clipping parameter',
                        default = 0.2)
    parser.add_argument('-s', '--skipping_steps', type=int, help='Number of steps for which control is fixed',
                        default = 1)
    parser.add_argument('--valNN_train_epoch', type=int, help='Number of epochs to train Value NN',
                        default = 10)
    parser.add_argument('--policyNN_train_epoch', type=int, help='Number of epochs to train Policy NN',
                        default = 3)
    args = parser.parse_args()

    main(network_id,  **vars(args))
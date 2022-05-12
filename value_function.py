"""
State-Value Function

Written by Patrick Coady (pat-coady.github.io)
Referring largely to https://github.com/mark-gluzman/MulticlassQueuingNetworkPolicyOptimization/blob/master/value_function.py
"""

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Embedding, Flatten 
from tensorflow.keras.regularizers import l2
from logger import Logger
import datetime

class NNValueFunction(tf.keras.Model):
    """ 
    Value neural network
    Input is observation and timestep.
    Ouput is the Value of the observation.
    """
    
    def __init__(self, obs_dim, hid1_mult, hid3_size, sz_voc, embed_dim, train_epoch, reg_str=5e-3):
        """
        obs_dim: dimension of the observation (input)
        hid1_mult: helps compute size of first hidden layer, hid1_mult*(obs_dim + embed_dim)
        hid3_size: size of the third hidden layer
        sz_voc: set to environment's horizon H, helps embed time component
        embed_dim: dimension of time's embedding output
        train_epoch: number of epochs when training
        reg_str: used for time embedding
        """
        super(NNValueFunction, self).__init__()
        # consume parameters
        self.obs_dim = obs_dim
        self.hid1_mult = hid1_mult
        self.hid3_size = hid3_size
        self.sz_voc = sz_voc  # H
        self.embed_dim = embed_dim
        self.reg_str = reg_str
        self.epochs = train_epoch
        self.lr = 1e-4 
        self.batch_size = 128
        
        # used to store previous step input
        self.replay_buffer_x = None
        self.replay_buffer_x_t = None
        self.replay_buffer_y = None

        self._build_graph()

    def _build_graph(self):
        """
        Construct Tensorflow NN architecture
        """
        # 1. Calculate each hidden layer's size
        hid1_size = (self.obs_dim + self.embed_dim) * self.hid1_mult
        hid3_size = self.hid3_size
        hid2_size = int(np.sqrt(hid1_size * hid3_size))

        # 2. Calculate each layer's kernel initializer's stddev
        # TODO: check whether our random_normal_initializer is necessary
        hid1_KIstd = np.sqrt(1 / (self.obs_dim + self.embed_dim))
        hid2_KIstd = np.sqrt(1 / hid1_size)
        hid3_KIstd = np.sqrt(1 / hid2_size)
        out_KIstd = np.sqrt(1 / hid3_size)

        # 3. Define all the layers
        # note that embedding's input_dim is the size of the dictionary, not the actual time component
        self.embed_layer = Embedding(input_dim=self.sz_voc, output_dim=self.embed_dim, \
            name='simple_embedding', embeddings_regularizer=l2(self.reg_str), trainable=True)
        self.flatten = Flatten()
        self.hidden_layer1 = Dense(hid1_size, activation='tanh', \
            kernel_initializer=tf.random_normal_initializer(stddev=hid1_KIstd))
        self.hidden_layer2 = Dense(hid2_size, activation='tanh', \
            kernel_initializer=tf.random_normal_initializer(stddev=hid2_KIstd))
        self.hidden_layer3 = Dense(hid3_size, activation='tanh', \
            kernel_initializer=tf.random_normal_initializer(stddev=hid3_KIstd))
        self.output_layer = Dense(1, activation=None, \
            kernel_initializer=tf.random_normal_initializer(stddev=out_KIstd))

    def call(self, x, x_t):
        """
        The forwardpass for our value function NN.
        x: input, the observation (state); has shape of self.obs_dim
        x_t: input, the time component when the state is observed; just an integer
        """
        embedded_time = self.embed_layer(x_t)
        embedded_time = self.flatten(embedded_time)
        output = self.hidden_layer1(tf.concat([x, embedded_time], axis=1))
        output = self.hidden_layer2(output)
        output = self.hidden_layer3(output)
        output = self.output_layer(output)
        return output

def fit_valueNN(model: NNValueFunction, x, x_t, y, logger: Logger):
    """
    Fit the value NN model given the training data from trajectories.
    Uses both current data and previous data (from previous update, previous outermost epoch).
    Args:
        model: the value function NN instance
        x: input, the observation (state); has shape of self.obs_dim
        x_t: input, the time component when the state is observed; just an integer
        y: label, the approximated value for the given input
        logger: logger class to save training loss and other stats
    """
    # 1. concatenate current data with previous data, and update the replay buffer
    x_train, x_t_train, y_train = None, None, None
    if model.replay_buffer_x is None:
        x_train, x_t_train, y_train = x, x_t, y
    else:
        x_train = np.concatenate([x, model.replay_buffer_x])
        x_t_train = np.concatenate([x_t, model.replay_buffer_x_t])
        y_train = np.concatenate([y, model.replay_buffer_y])
    model.replay_buffer_x = x  
    model.replay_buffer_x_t = x_t
    model.replay_buffer_y = y  

    # 2. prepare loss, optimizer, and dataloader
    optimizer = tf.keras.optimizers.Adam(model.lr) #TODO: see what to do with learning rate
    # TODO: make sure len(x_train) gives the actual size
    dataloader = tf.data.Dataset.from_tensor_slices((x_train, x_t_train, \
        y_train)).shuffle(x_train.shape[0]).batch(model.batch_size)

    for e in range(model.epochs):
        for x_train_i, x_t_train_i, y_train_i in dataloader:
            with tf.GradientTape() as tape:
                y_hat_i = model(x_train_i, x_t_train_i, training=True)
                loss = tf.keras.losses.mean_squared_error(y_train_i, y_hat_i)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    logger.log({'ValFuncLoss': loss})  # loss from last epoch
    print(loss)

def test():
    now = datetime.datetime.utcnow().strftime("%b-%d_%H-%M-%S")  # create unique directories
    time_start= datetime.datetime.now()
    logger = Logger('dummy', now=now, time_start=time_start)

    val_func_model = NNValueFunction(obs_dim=400, hid1_mult=1, hid3_size=5, sz_voc=360, embed_dim=6, train_epoch=10, reg_str=5e-3)
    dummy_x = tf.cast(np.resize(np.arange(0,400*3), (3,400)), tf.float32)
    dummy_x_t = tf.cast(np.resize(np.arange(3), (3,1)), tf.float32)
    dummy_y = tf.cast(np.resize(np.arange(3), (3,1)), tf.float32)

    fit_valueNN(val_func_model, dummy_x, dummy_x_t, dummy_y, logger)

if __name__ == '__main__':
    test()
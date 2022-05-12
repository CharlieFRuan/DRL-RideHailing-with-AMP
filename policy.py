import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Flatten 
from tensorflow.keras.regularizers import l2
import tensorflow_probability as tfp
import datetime
from logger import Logger

class NNPolicy(tf.keras.Model):
    """ 
    Policy neural network
    Input is observation and timestep.
    Ouput is probability distribution over all actions.
    """

    def __init__(self, obs_dim, act_dim, hid1_mult, hid3_size, sz_voc, embed_dim, train_epoch=3, \
        reg_str=5e-3, kl_targ=np.inf, clipping_range=0.2, temp=2.0):
        """
        obs_dim: dimension of the observation (input)
        act_dim: total number of different actions (R*R types of trips in this case)
        hid1_mult: helps compute size of first hidden layer, hid1_mult*(obs_dim + embed_dim)
        hid3_size: size of the third hidden layer
        sz_voc: set to environment's horizon H, helps embed time component
        embed_dim: dimension of time's embedding output
        train_epoch: number of epochs when training
        reg_str: used for time embedding
        kl_targ: D_KL target value
        clipping_range: Initial clipping parameter
        temp: temperature parameter
        """
        super(NNPolicy, self).__init__()
        # consume parameters
        self.obs_dim = obs_dim  # not including time component
        self.act_dim = act_dim
        self.hid1_mult = hid1_mult
        self.hid3_size = hid3_size
        self.sz_voc = sz_voc
        self.embed_dim = embed_dim
        self.reg_str = reg_str

        # PPO specific parameters
        self.kl_targ = kl_targ  # A large KL target implies no early stopping
        self.clipping_range = clipping_range
        self.temp = temp

        # training related
        self.train_epoch = train_epoch
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.lr = 5e-4 

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
        self.act_prob_output_layer = Dense(self.act_dim, activation='softmax', \
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
        output = self.act_prob_output_layer(output)

        return output

def updatePolicy(model: NNPolicy, observes, times, actions, advantages, logger: Logger):
    """
    Policy Neural Network update using data from trajectories. 
    Referred to Machine Learning with Phil's PPO implementation.
    :param observes: states
    :param times: time components
    :param actions: actions
    :param advantages: estimation of antantage function at observed states
    :param logger: statistics accumulator
    """
    # 1. get the probability to take the same actions (from the trajectories) with the current policy
    old_act_output = model(observes, times)
    old_distr = tfp.distributions.Categorical(old_act_output)
    old_probs = old_distr.log_prob(actions)

    optimizer = tf.keras.optimizers.Adam(model.lr)

    # 2. Train
    for e in range(model.train_epoch):
        with tf.GradientTape() as tape:
            #TODO: see if tf.convert_to_tensor is needed
            # 2.1 Calculate the new probabilities
            new_act_output = model(observes, times, training=True)
            new_distr = tfp.distributions.Categorical(new_act_output)
            new_probs = new_distr.log_prob(actions)

            # 2.2 Calculate PPO's surrogate objective function
            prob_ratio = tf.math.exp(new_probs - old_probs)
            weighted_probs = advantages * prob_ratio
            clipped_probs = tf.clip_by_value(prob_ratio, 1-model.clipping_range, 1+model.clipping_range)
            weighted_clipped_probs = clipped_probs * advantages
            loss = -tf.math.minimum(weighted_probs, weighted_clipped_probs)
            loss = tf.math.reduce_mean(loss)

            # 2.3 backpropagate
            model_params = model.trainable_variables
            gradients = tape.gradient(loss, model_params)
            optimizer.apply_gradients(zip(gradients, model_params))
    
    print(loss)
    logger.log({'PolicyLoss': loss,
                'Clipping': model.clipping_range,
                '_lr_multiplier': model.lr_multiplier})


def test():
    now = datetime.datetime.utcnow().strftime("%b-%d_%H-%M-%S")  # create unique directories
    time_start= datetime.datetime.now()
    logger = Logger('dummy', now=now, time_start=time_start)

    policy_model = NNPolicy(obs_dim=400, act_dim=25, hid1_mult=1, hid3_size=5, sz_voc=360, \
        embed_dim=6, train_epoch=10, reg_str=5e-3)
    
    # dummy variables for testing
    x = tf.cast(np.resize(np.arange(0,400*3), (3,400)), tf.float32)
    x_t = tf.cast(np.resize(np.arange(3), (3,1)), tf.float32)
    actions = tf.cast(np.resize(np.arange(3), (3,1)), tf.float32)
    advantages = tf.cast(np.resize(np.arange(3), (3,1)), tf.float32)

    updatePolicy(policy_model, x, x_t, actions, advantages, logger)

if __name__ == '__main__':
    test()
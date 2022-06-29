import numpy as np
from agent import Agent


class Env(object):
    """ The transportation network. """
    network_name = 'transportation_network' # Charlie 5/3/22 to make the code run
    
    R = 5  # number of regions
    N = 50  # number of cars
    H = 20  # horizon in minute (10 now for testing?)
    num_slots = 3
    len_slot = 120  # in minute

    # Actual scale below
    # R = 5  # number of regions
    # N = 100  # number of cars
    # H = 360  # horizon in minute
    # num_slots = 3 # we split the entire day horizon into 3 slots (each slot is 2 hours) just like in the paper
    # len_slot = 120  # in minute


    # Passenger arrival rate (number per minute)
    # This decides the car initial distribution
    # in paper, it is number of passenger arrivals at region o in a minute (here we scale by car)
    lambda_by_region = np.asarray([[0.108, 0.108, 0.108, 0.108, 1.08], \
                                   [0.72, 0.48, 0.48, 0.48, 0.12], \
                                   [0.12, 0.12, 0.12, 1.32, 0.12]])  # per car, per hour
    lambda_by_region = lambda_by_region * N / 60.  # all cars, per minute
    lambda_max = lambda_by_region.max(axis=0)  # max over time
    accept_prob = lambda_by_region / lambda_max[np.newaxis, :]
    # for converting time-inhomogeneous Poisson process into a time-homogeneous one
    dest_prob = np.asarray([
        [[.6, .1, 0, .3, 0], \
         [.1, .6, 0, .3, 0], \
         [0, 0, .7, .3, 0], \
         [.2, .2, .2, .2, .2], \
         [.3, .3, .3, .1, 0]], \
        [[.1, 0, 0, .9, 0], \
         [0, .1, 0, .9, 0], \
         [0, 0, .1, .9, 0], \
         [.05, .05, .05, .8, .05], \
         [0, 0, 0, .9, .1]], \
        [[.9, .05, 0, .05, 0], \
         [.05, .9, 0, .05, 0], \
         [0, 0, .9, .1, 0], \
         [.3, .3, .3, .05, .05], \
         [0, 0, 0, .1, .9]]
    ])
    num_imp_ride = np.zeros(R * R, dtype=int)  # number of impossible ride request type
    __ct = 0
    for o in range(R):
        for d in range(R):
            if np.all(lambda_by_region[:, o] * dest_prob[:, o, d] == 0):
                __ct += 1
            num_imp_ride[o * R + d] = __ct
    L = 5  # candidate patience time in minute

    # Mean travel time in minute
    tau = np.asarray([[[0.15, 0.25, 1.25, 0.2, 0.4], \
                       [0.25, 0.1, 1.1, 0.1, 0.3], \
                       [1.25, 1.1, 0.1, 1, 0.65], \
                       [0.25, 0.15, 1, 0.15, 0.25], \
                       [0.5, 0.4, 0.75, 0.25, 0.2]], \
                      [[0.15, 0.25, 1.25, 0.2, 0.4], \
                       [0.25, 0.1, 1.1, 0.1, 0.3], \
                       [1.25, 1.1, 0.1, 1, 0.65], \
                       [0.2, 0.1, 1, 0.15, 0.25], \
                       [0.4, 0.3, 0.65, 0.25, 0.2]], \
                      [[0.15, 0.25, 1.25, 0.2, 0.4], \
                       [0.25, 0.1, 1.1, 0.1, 0.3], \
                       [1.25, 1.1, 0.1, 1, 0.65], \
                       [0.2, 0.1, 1, 0.15, 0.25], \
                       [0.4, 0.3, 0.65, 0.25, 0.2]]])
    tau = (np.ceil(np.array(tau) * 60)).astype('int') # convert to minutes
    tau_max = tau.max(axis=1).max(axis=0)  # max travel time to a region, for each region; hence array R in length
    car_dims = 1 + tau_max + L  # the second dimension in s_c in paper; + 1 is for zero; dim for each region
    car_dims_cum = np.insert(car_dims.cumsum(), 0, 0) # the last term [-1] would be the dimension for flattened s_c
    stay_empty_nby_car_dim = (1 + L) * R # dimension of s_l in paper
    # Nearby cars associated with each region, assigned to "stay empty" upon reaching destination
    # so its region will not change, neither will its total distance to the region;
    # before reaching the destination, this car is not free to be assigned a trip anymore!
    car_dim = car_dims_cum[-1] + stay_empty_nby_car_dim # combine s_c and s_l into one vector
    psg_dim = R * R - num_imp_ride[-1] # s_p in paper, but remove the number of types of rides that will not be required
    obs_dim = car_dim + psg_dim  # Observation dimension excluding time component

    # Rewards (these 2 variable almost never used for now; since we assume either 1 or 0)
    # Car-passenger matching
    c = np.ones((num_slots, R, R), dtype=float)
    # Empty-car routing
    tilde_c = np.zeros((num_slots, R, R), dtype=float)
    for o in range(R):
        assert np.all(tilde_c[:, o, o] == 0), 'Rewards for idling-at-destination action must be zero!'

    def __init__(self):

        # Ride requests
        self.queue = None  # all day
        self.next_ride = None  # index
        # self.all_rewards = None  # total number of ride requests over the horizon
        self.car_init_dist = None

    @staticmethod
    def min_to_slot(minute):
        """ Minute to slot
         minute in [0, H)
         Passenger arrivals, etc., by minute 
         Convert envent time (continuous) to slot
         """

        return min(int(minute / Env.len_slot), Env.num_slots - 1)
        # min((minute - 1) // Env.len_slot, Env.num_slots - 1)

    def reset(self, dest_known=True):
        """ Reset everything at the start of an episode """

        # Ride requests
        queue_by_region = [[] for r in range(Env.R)] # Each list is the arrival times of passengers
        for r in range(Env.R):
            # Generate arrival for each region
            tpast = 0
            while True:
                # Simulating Poisson Process arrival
                interarriv = np.random.exponential(scale=1. / Env.lambda_max[r], size=None) 
                tpast += interarriv
                if tpast >= Env.H:
                    break
                queue_by_region[r].append(tpast)
        queue_inhomo = [[] for r in range(Env.R)] # perform accept rejection for simulating arrivals
        for r in range(Env.R):
            for event in queue_by_region[r]:
                idx = Env.min_to_slot(event)
                if np.random.choice(a=2, size=None, p=[1 - Env.accept_prob[idx, r], Env.accept_prob[idx, r]]) > 0.5:
                    queue_inhomo[r].append(event)
        del queue_by_region
        # List of events (all regions), each entry is 3-element list
        # e.g. [17.74168623928455, 4, 1], denoting arrival time, origin, destination
        self.queue = [] 
        if dest_known:
            for r in range(Env.R):
                for event in queue_inhomo[r]:
                    idx = Env.min_to_slot(event)
                    self.queue.append([event, r, np.random.choice(a=Env.R, size=None, p=Env.dest_prob[idx, r])])
        else:
            for r in range(Env.R):
                for event in queue_inhomo[r]:
                    self.queue.append([event, r])
        self.queue.sort(key=lambda x: x[0])  # sort by arrival time
        self.next_ride = 0

        # Generate car initial distribution based on first time slot's (first two hours) passenger arrival rate
        init_car_dist = Env.lambda_by_region[0].copy()
        init_car_dist = init_car_dist / init_car_dist.sum() * Env.N
        surplus = init_car_dist - init_car_dist.astype('int')
        surplus_from_largest = surplus.argsort()[::-1]
        total_surplus = int(surplus.sum())

        init_car_dist = init_car_dist.astype('int')
        i, j = 0, 0  # sliding window
        while i < len(surplus) and total_surplus > 0:
            while j < len(surplus) and abs(surplus[surplus_from_largest[j]] - surplus[surplus_from_largest[i]]) < 1e-8:
                j += 1
            probs = np.asarray([self.lambda_by_region[0][surplus_from_largest[k]] for k in range(i, j)])
            probs = probs / probs.sum()
            remaining_cars = np.random.multinomial(min(j - i, total_surplus), probs)
            for k in range(i, j):
                init_car_dist[surplus_from_largest[k]] += remaining_cars[k - i]
            total_surplus -= min(j - i, total_surplus)
        self.car_init_dist = init_car_dist

    def get_initial_state(self):
        """ Get initial state """

        if self.car_init_dist is None:
            self.reset()  # Sometimes self.car_init_dist is not None but still resetting maybe needed!
        s_cp = np.zeros(self.obs_dim, dtype=int)  # unscaled; s_cp contains s_c, s_l, and s_p
        for reg in range(self.R):
            # For each region, we use car_dims_cum to determine the start of that region's status
            # in s_cp (for instance, region 1 starts at 0, since car_dims_cum[0] = 0).
            # This for loop initialize the number of cars 0 minute away from each region, which is
            # simply the initial distribution of cars.
            s_cp[self.car_dims_cum[reg]] = self.car_init_dist[reg]
        # Now we initialize passengers' status in s_cp
        s_t = 0
        while self.next_ride < len(self.queue) and (self.queue[self.next_ride][0] < s_t + 1):
            # Stop when run out of events, or when finished processing events before current time (s_t); 
            # note that s_t is in granularity of a whole minute (e.g. arrival at t=0.332 is grouped to minute 0);
            # hence the < s_t + 1.
            # For trip_type, o*R+d gives us the relative index of the trip in s_cp
            trip_type = self.queue[self.next_ride][1] * self.R + self.queue[self.next_ride][2] 
            # Reminder: car_dim is the combined length of s_c and s_l; below gives us the actual index in s_cp
            s_cp[self.car_dim + trip_type-self.num_imp_ride[trip_type]] += 1
            self.next_ride += 1
        return s_cp, s_t

    def num_avb_nby_cars(self, state_vector):
        """ 
        Number of available (a.k.a. free) nearby cars 
        Return total number of cars within L minutes away from their destination.
        i.e. Return the I_t in paper.
        """

        num = 0
        for reg in range(self.R):
            # Again, car_dims_cum[reg] marks the start of the region's s_c in s_cp (state_vector)
            # state_vector[self.car_dims_cum[reg]]: number of cars 0 minute till arriving at region reg
            # state_vector[self.car_dims_cum[reg]] + self.L: number of cars L minutes till arriving at region reg
            # Only look at 0 to L minutes because that is passengers' patience time
            num += state_vector[self.car_dims_cum[reg]:(self.car_dims_cum[reg] + self.L + 1)].sum()
        return num

    def get_next_state(self, s_post, dec_epoch):
        """ Get the next state from the post-decision state of the last candidate
        s_post updated in-place
        """
        s_post[self.car_dim:] = 0  # Passengers unattended leave the transportation network.
        next_dec_epoch = dec_epoch + 1
        # print('Next decision epoch: ', next_dec_epoch)
        if next_dec_epoch < self.H:
            # Passengers accumulation.
            while self.next_ride < len(self.queue) and self.queue[self.next_ride][0] < next_dec_epoch + 1:
                # Read in arrival events from queue, allocate them to s_cp's s_p portion
                # print('Next ride index: ', self.next_ride)
                # print('Time of arrival: ', self.queue[self.next_ride][0])
                trip_type = self.queue[self.next_ride][1] * self.R + self.queue[self.next_ride][2]
                s_post[self.car_dim + trip_type - self.num_imp_ride[trip_type]] += 1
                self.next_ride += 1
            # Car dynamics.
            # 1) Unavailable cars -> available; that is, move all cars in s_l to the corresponding s_c
            # since s_l is just to prevent those cars being matched in that single SDM
            for reg in range(self.R):
                s_post[self.car_dims_cum[reg]:(self.car_dims_cum[reg]+self.L+1)] += \
                    s_post[(self.car_dims_cum[-1] + reg * (1 + self.L)):
                           (self.car_dims_cum[-1] + (reg+1) * (1 + self.L))] # LHS is s_c, RHS is s_l
                s_post[(self.car_dims_cum[-1] + reg * (1 + self.L)):
                       (self.car_dims_cum[-1] + (reg + 1) * (1 + self.L))] = 0 # clear out s_l

            # 2) Available cars after one time interval; that is, shift s_c forward
            for reg in range(self.R): 
                # For idling cars, it accumulates (since idling cars keep being ideled), hence the +=
                # instead of shifting like others
                s_post[self.car_dims_cum[reg]] += s_post[self.car_dims_cum[reg] + 1]
            for reg in range(self.R):
                s_post[(self.car_dims_cum[reg] + 1):(self.car_dims_cum[reg + 1] - 1)] = \
                    s_post[(self.car_dims_cum[reg] + 2):self.car_dims_cum[reg + 1]]
            for reg in range(self.R):
                s_post[self.car_dims_cum[reg + 1] - 1] = 0
        else:  # next_dec_epoch == self.H
            s_post[:] = -1  # terminal state

        return next_dec_epoch


def test():
    env = Env()
    # print('Number impossible ride request type: \n', env.num_imp_ride)
    # print('Car dim: \n', env.car_dim)
    # print('State vector dimension: \n', env.obs_dim)
    # print()
    # s_cp_init, s_t_init = env.get_initial_state()
    # print('Initial car-passenger state: \n', s_cp_init[env.car_dim:])
    # print()
    # print('To do (tentative): \n'
    #       '1. State; 2. Transition model; 3. Processing a candidate given action probabilities from policy network;')
    
    agent = Agent(network=env, weights=None, scaler=None, hid1_mult=1, hid3_size=5, sz_voc=env.H, embed_dim=env.num_slots*2, model_dir=None, cur_iter=0)
    agent.run_episode()

if __name__ == '__main__':
    test()

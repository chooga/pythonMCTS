import argparse
import os

import gym
import time
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import gym.spaces as spaces
import random

import matplotlib.pyplot as plt

UP_ACTION = 2
DOWN_ACTION = 3
NO_OP_ACTION = 0
constrained_actions = [NO_OP_ACTION, UP_ACTION, DOWN_ACTION]

class Model(): #https://github.com/tmoer/alphazero_singleplayer/blob/db742bcbd61e1d62a6958136ca7bb2ae11053971/alphazero.py
    def __init__(self, Env, lr, n_hidden_layers, n_hidden_units):
        # Check the Gym environment
        self.action_dim, self.action_discrete = check_space(Env.action_space)
        self.state_dim, self.state_discrete = check_space(Env.observation_space)
        if not self.action_discrete:
            raise ValueError('Continuous action space not implemented')

        # Placeholders
        if not self.state_discrete:
            self.x = x = tf.placeholder("float32", shape=np.append(None, self.state_dim), name='x')  # state
        else:
            self.x = x = tf.placeholder("int32", shape=np.append(None, 1))  # state
            x = tf.squeeze(tf.one_hot(x, self.state_dim, axis=1), axis=2)

        x = tf.layers.flatten(x)

        # Feedforward: Can be modified to any representation function, e.g. convolutions, residual networks, etc.
        for i in range(n_hidden_layers):
            x = slim.fully_connected(x, n_hidden_units, activation_fn=tf.nn.elu)

        # Output
        log_pi_hat = slim.fully_connected(x, 3, activation_fn=None)  #TODO second argument self.action_dim
        self.pi_hat = tf.nn.softmax(log_pi_hat)  # policy head
        self.V_hat = slim.fully_connected(x, 1, activation_fn=None)  # value head


        # Loss
        self.V = tf.placeholder("float32", shape=[None, 1], name='V')
        self.pi = tf.placeholder("float32", shape=[None, self.action_dim], name='pi')
        self.V_loss = tf.losses.mean_squared_error(labels=self.V, predictions=self.V_hat)

        self.pi_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.pi, logits=log_pi_hat)
        self.loss = self.V_loss + tf.reduce_mean(self.pi_loss)

        self.lr = tf.Variable(lr, name="learning_rate", trainable=False)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
        self.train_op = optimizer.minimize(self.loss)

    def train(self, sb, Vb, pib):
        self.sess.run(self.train_op, feed_dict={self.x: preprocess(sb),
                                                self.V: Vb,
                                                self.pi: pib})

    def predict_V(self, s):
        return self.sess.run(self.V_hat, feed_dict={self.x: preprocess(s)})

    def predict_pi(self, s):
        return self.sess.run(self.pi_hat, feed_dict={self.x: preprocess(s)})

class Database():

    def __init__(self,max_size,batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.experiences = []
        self.insert_index = 0
        self.size = 0
        self.sample_array = None
        self.sample_index = 0

    def clear(self):
        self.experiences = []
        self.insert_index = 0
        self.size = 0

    def store(self,experience):
        if self.size < self.max_size:
            self.experiences.append(experience)
            self.size +=1
        else: #the next cell will be rewritten
            self.experiences[self.insert_index] = experience
            self.insert_index += 1
            if self.insert_index >= self.size:
                self.insert_index = 0

    def reshuffle(self):
        self.sample_array = np.arange(self.size)
        random.shuffle(self.sample_array)
        self.sample_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if (self.sample_index + self.batch_size > self.size) and (not self.sample_index == 0):
            self.reshuffle()  # Reset for the next epoch
            raise (StopIteration)

        if (self.sample_index + 2 * self.batch_size > self.size):
            indices = self.sample_array[self.sample_index:]
            batch = [self.experience[i] for i in indices]
        else:
            indices = self.sample_array[self.sample_index:self.sample_index + self.batch_size]
            batch = [self.experience[i] for i in indices]
        self.sample_index += self.batch_size

        arrays = []
        for i in range(len(batch[0])):
            to_add = np.array([entry[i] for entry in batch])
            arrays.append(to_add)
        return tuple(arrays)

    next = __next__


class State():
    ''' State object '''

    def __init__(self, index, r, terminal, parent_action, na, model):
        ''' Initialize a new state '''
        self.index = index  # state
        self.r = r  # reward upon arriving in this state
        self.terminal = terminal  # whether the domain terminated in this state
        self.parent_action = parent_action
        self.n = 0
        self.model = model

        self.evaluate()
        # Child actions
        self.na = na
        self.child_actions = [Action(a+1, parent_state=self, Q_init=self.V) for a in range(na)] #TODO constrained actionspace "+1" added
        self.priors = model.predict_pi(index[None,]).flatten()
 #       self.priors = np.ones(len(self.child_actions))

    def select(self, c=1.0): #alternativ value 2.5 or 1.0
        ''' Select one of the child actions based on UCT rule '''
        UCT = np.array(
            [child_action.Q + prior * c * (np.sqrt(self.n + 1) / (child_action.n + 1)) for child_action, prior in
             zip(self.child_actions, self.priors)])
        winner = argmax(UCT)
        return self.child_actions[winner]

    def evaluate(self):
        ''' Bootstrap the state value '''
        self.V = np.squeeze(self.model.predict_V(self.index[None,])) if not self.terminal else np.array(0.0)
        #self.V = self.r

    def update(self):
        ''' update count on backward pass '''
        self.n += 1


class Action():
    ''' Action object '''

    def __init__(self, index, parent_state, Q_init=0.0):
        self.index = index
        self.parent_state = parent_state
        self.W = 0.0
        self.n = 0
        self.Q = Q_init

    def add_child_state(self, s1, r, terminal, model):
        self.child_state = State(s1, r, terminal, self, self.parent_state.na, model)
        return self.child_state

    def update(self, R):
        self.n += 1
        self.W += R
        self.Q = self.W / self.n

class MCTS():
    ''' MCTS object '''

    def __init__(self, root, root_index, model, na, gamma):
        self.root = None
        self.root_index = root_index
        self.model = model
        self.na = na
        self.gamma = gamma

    def search(self, n_mcts, c, env, mcts_env, skip_frame):
        ''' Perform the MCTS search from the root '''
        if self.root is None:
            self.root = State(self.root_index, r=0.0, terminal=False, parent_action=None, na=self.na,
                              model=self.model)  # initialize new root
        else:
            self.root.parent_action = None  # continue from current root
        if self.root.terminal:
            raise (ValueError("Can't do tree search from a terminal state"))

        env = getBaseEnv(env)
        snapshot = env.clone_full_state()  # for Atari: snapshot the root at the beginning

        for i in range(n_mcts):
            state = self.root  # reset to root for new trace
            env.restore_full_state(snapshot)
            r = 0

            while not state.terminal:
                action = state.select(c=c)
                for frame in range(skip_frame):
                    s1, r1, t, _ = mcts_env.step(action.index)
                    r += r1
                r /= skip_frame
                if hasattr(action, 'child_state'):
                    state = action.child_state  # select
                    continue
                else:
                    state = action.add_child_state(s1, r, t, self.model)  # expand
                    break

            # Back-up
            R = state.V
            while state.parent_action is not None:  # loop back-up until root is reached
                R = state.r + self.gamma * R
                action = state.parent_action
                action.update(R)
                state = action.parent_state
                state.update()

    def forward(self, a, s1):
        ''' Move the root forward '''
        index_diff = np.linalg.norm(self.root.child_actions[a].child_state.index - s1)
        if not hasattr(self.root.child_actions[a], 'child_state'):
            self.root = None
            self.root_index = s1
        elif np.linalg.norm(self.root.child_actions[a].child_state.index - s1) > 0.01:
            print('Warning: this domain seems stochastic. Not re-using the subtree for next search. ' +
                  'To deal with stochastic environments, implement progressive widening.')
#            time.sleep(2)
            self.root = None
            self.root_index = s1
        else:
            self.root = self.root.child_actions[a].child_state

    def return_results(self, temp):
        ''' Process the output at the root node '''
        counts = np.array([child_action.n for child_action in self.root.child_actions])
        Q = np.array([child_action.Q for child_action in self.root.child_actions])
        pi_target = stable_normalizer(counts, temp)
        V_target = np.sum((counts / np.sum(counts)) * Q)[None]
        return self.root.index, pi_target, V_target

#helpers-methods
def check_space(space):
    ''' Check the properties of an environment state or action space '''
    if isinstance(space,spaces.Box):
        dim = space.shape
        discrete = False
    elif isinstance(space,spaces.Discrete):
        dim = space.n
        discrete = True
    else:
        raise NotImplementedError('This type of space is not supported')
    return dim, discrete

def getBaseEnv(env):
    if type(env) == gym.wrappers.time_limit.TimeLimit:
        env = env.env
    while hasattr(env, 'env'):
        env = env.env
    return env

def argmax(x):
    ''' assumes a 1D vector x '''
    x = x.flatten()
    if np.any(np.isnan(x)):
        print('Warning: Cannot argmax when vector contains nans, results will be wrong')
    try:
        winners = np.argwhere(x == np.max(x)).flatten()
        winner = random.choice(winners)
    except:
        winner = np.argmax(x) # numerical instability ?
    return winner

def store_safely(folder,name,to_store):
    ''' to prevent losing information due to interruption of process'''
    new_name = folder+name+'.npy'
    old_name = folder+name+'_old.npy'
    if os.path.exists(new_name):
        import shutil
        shutil.copyfile(new_name,old_name)
    np.save(new_name,to_store)
    if os.path.exists(old_name):
        os.remove(old_name)

def symmetric_remove(x,n):
    ''' removes n items from beginning and end '''
    odd = bool(n & 1) #true if number is odd, false else
    half = int(n/2)
    if half > 0:
        x = x[half:-half]
    if odd:
        x = x[1:]
    return x

def stable_normalizer(x,temp):
    ''' Computes x[i]**temp/sum_i(x[i]**temp) '''
    x = (x / np.max(x))**temp
    return np.abs(x/np.sum(x))

def preprocess(I): #https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    return I
    I = I[35:195]  # crop
    I = I[0::2, 0::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    print(I)
    print('the observation has the {} times {}'.format(len(I), len(I[0])))
    #return I.astype(np.float).ravel()

#class PlaningModel(Env=env, lr=lr, n_hidden_layers=n_hidden_layers):



def MCTSAgent(game,n_ep,n_mcts,max_ep_len,lr,c,gamma,data_size,batch_size,temp,n_hidden_layers,n_hidden_units, skip_frame):
    episode_returns = []  # storage
    timepoints = []
    # Environments
    env = gym.make('Pong-v0')
    mctsEnv = gym.make('Pong-v0')
    env = getBaseEnv(env)
    mctsEnv = getBaseEnv(mctsEnv)

    D = Database(max_size=data_size, batch_size=batch_size)
    model = Model(Env=env, lr=lr, n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units)
    t_total = 0  # total steps
    R_best = -np.Inf

    with tf.Session() as sess:
        model.sess = sess
        sess.run(tf.global_variables_initializer())
        for ep in range(n_ep):
            start = time.time()
            s = env.reset()
            R = 0.0  # Total return counter
            a_store = []
            seed = np.random.randint(1e7)  # draw some Env seed
            env.seed(seed)
            mctsEnv.reset()
            mctsEnv.seed(seed)

            mcts = MCTS(root_index=s, root=None, model=model, na = len(constrained_actions), gamma=gamma)  # the object responsible for MCTS searches TODO #na=model.action_dim
            for t in range(max_ep_len):
                # MCTS step
                mcts.search(n_mcts=n_mcts, c=c, env=env, mcts_env=mctsEnv,skip_frame=skip_frame)  # perform a forward search
                state, pi, V = mcts.return_results(temp)  # extract the root output
                D.store((state, V, pi))

                # Make the true step
                a = np.random.choice(len(pi), p=pi)
                a_store.append(a)
                s1, r, terminal, _ = env.step(a)
                for _ in range(skip_frame-1):
                    s1, r, terminal, _ = env.step(a)
                    R += r
                    if terminal:
                        break
                R += r
                R /= skip_frame
                t_total += n_mcts  # total number of environment steps (counts the mcts steps)

                if terminal:
                    break
                else:
                    mcts.forward(a, s1)

            # Finished episode
            episode_returns.append(R)  # store the total episode return
            timepoints.append(t_total)  # store the timestep count of the episode return
            store_safely(os.getcwd(), 'result', {'R': episode_returns, 't': timepoints})

            if R > R_best:
                a_best = a_store
                seed_best = seed
                R_best = R
            print('Finished episode {}, total return: {}, total time: {} sec'.format(ep, np.round(R, 2),
                                                                                     np.round((time.time() - start), 1)))
            # # Train
            # D.reshuffle()
            # for epoch in range(1):
            #     for sb, Vb, pib in D:
            #         model.train(sb, Vb, pib)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='CartPole-v0', help='Training environment')
    parser.add_argument('--n_ep', type=int, default=500, help='Number of episodes')
    parser.add_argument('--n_mcts', type=int, default=40, help='Number of MCTS traces per step')
    parser.add_argument('--max_ep_len', type=int, default=300, help='Maximum number of steps per episode')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--c', type=float, default=1.5, help='UCT constant')
    parser.add_argument('--temp', type=float, default=1.0,
                        help='Temperature in normalization of counts to policy target')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount parameter')
    parser.add_argument('--data_size', type=int, default=1000, help='Dataset size (FIFO)')
    parser.add_argument('--batch_size', type=int, default=32, help='Minibatch size')
    parser.add_argument('--window', type=int, default=25, help='Smoothing window for visualization')

    parser.add_argument('--n_hidden_layers', type=int, default=2, help='Number of hidden layers in NN')
    parser.add_argument('--n_hidden_units', type=int, default=128, help='Number of units per hidden layers in NN')
    parser.add_argument('--skip_frame', type=int, default=3, help='Number of units per hidden layers in NN')

    args = parser.parse_args()

    episode_returns,timepoints,a_best,seed_best,R_best = MCTSAgent(game=args.game,n_ep=args.n_ep,n_mcts=args.n_mcts,
                                        max_ep_len=args.max_ep_len,lr=args.lr,c=args.c,gamma=args.gamma,
                                        data_size=args.data_size,batch_size=args.batch_size,temp=args.temp,
                                        n_hidden_layers=args.n_hidden_layers,n_hidden_units=args.n_hidden_units,skip_frame=args.skip_frame)

    fig,ax = plt.subplots(1,figsize=[7,5])
    total_eps = len(episode_returns)
    episode_returns = np.convolve(episode_returns, np.ones(args.window)/args.window, mode='valid')
    ax.plot(symmetric_remove(np.arange(total_eps),args.window-1),episode_returns,linewidth=4,color='darkred')
    ax.set_ylabel('Return')
    ax.set_xlabel('Episode',color='darkred')
    plt.savefig(os.getcwd()+'/learning_curve.png',bbox_inches="tight",dpi=300)
# for episode in range(10):
#     obs = env.reset()
#     for step in range(50):
#         action = env.action_space.sample()  # or given a custom model, action = policy(observation)
#         observation, reward, done, info = env.step(action)
#         env.render()
#         if(done):
#             break
#         time.sleep(0.1)
#
# env.close()
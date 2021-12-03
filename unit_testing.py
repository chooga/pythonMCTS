import argparse
import os

import gym
import time
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import numpy as np
import gym.spaces as spaces
import random
from PIL import Image
from main import Database, getBaseEnv, MCTS, applyNoise, check_space, preprocess, Model
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

#this unit test is about giving the agent a reproducable environment and see whether the agent is capable of blocking the first hit
#experiments has shown, that at about 32 frames into the env, the enemy hits the ball
#at around frame 39 the ball passes the middle point
#and after 50 the first negativ reward can be recieved, if the ball is not blocked


eps = np.finfo(np.float32).eps.item()

def model():
    action_dim = 3
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 4,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, padding="valid", activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(action_dim, activation="linear")(layer5)

    return tf.keras.Model(inputs=inputs, outputs=action)


def save_image(s1, ep):
    newimage = Image.new('RGB', (len(s1[0]), len(s1)))  # type, size
    newimage.putdata([tuple(p) for row in s1 for p in row])
    newimage.save("filename_{}.png".format(ep))  # takes type from filename extension


def main(game,n_ep,n_mcts,max_ep_len,lr,c,gamma,data_size,batch_size,temp,n_hidden_layers,n_hidden_units, skip_frame):

    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.1  # Minimum epsilon greedy parameter
    epsilon_max = 1.0  # Maximum epsilon greedy parameter
    epsilon_interval = (
            epsilon_max - epsilon_min
    )
    preDefinedActions = []
    for _ in range(50):
       preDefinedActions.append(1)
#    for _ in range(30):
#        preDefinedActions.append(2)
#        preDefinedActions.append(5)

    episode_returns = []  # storage
    timepoints = []
    # Environments
    env = gym.make('Pong-ram-v0')
    mctsEnv = gym.make('Pong-ram-v0')
    env = getBaseEnv(env)
    mctsEnv = getBaseEnv(mctsEnv)
    print(len(preDefinedActions))

    D = Database(max_size=data_size, batch_size=batch_size)

#    model1 = model()

#    model_target = model(env)
    model = Model(Env=env, lr=lr, n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units)

    t_total = 0  # total steps
    R_best = -np.Inf
    with tf.Session(config=tf.ConfigProto()) as sess:
        model.sess = sess
        sess.run(tf.global_variables_initializer())
        for ep in range(n_ep):
            start = time.time()
            s = env.reset()
            R = 0.0  # Total return counter
            a_store = []
            seed = np.random.randint(1e7)  # draw some Env seed
            seed = 2566437
            env.seed(seed)
            mctsEnv.reset()
            mctsEnv.seed(seed)

            mcts = MCTS(root_index=s, root=None, model=model, na=3,
                    gamma=gamma)  # the object responsible for MCTS searches TODO #na=model.action_dim
            for t in range(max_ep_len):
            # MCTS step
                if t < len(preDefinedActions):
                    s1, r1, timePassed, _ = env.step(preDefinedActions[t])
    #               env.render("human")
    #               print(t)
    #               print(preDefinedActions[t])
                    continue
                mcts.search(n_mcts=n_mcts, c=c, env=env, mcts_env=mctsEnv,
                                skip_frame=skip_frame)  # perform a forward search
                state, pi, V = mcts.return_results(temp)  # extract the root output

                pi = applyNoise(pi)
                D.store((state, V, pi))
    # Make the true step
                a = np.random.choice(len(pi), p=pi)
                a_store.append(a + 1)
                #                 s1, r, terminal, _ = env.step(a+1)
                # #                env.render("human")
                # #                if (r > 0):
                # #                    input("waiting")
                #                 R += r
                for skfr in range(skip_frame):
                    s1, r, terminal, _ = env.step(a + 1)
                        #if (r > 0):
                    #        input("waiting")
                    R += r
                    env.render("human")
#                    print("the move applied was {}, while the pi was {}".format(a+1, pi))
                    if terminal:
                        break
                else:
                    continue
                t_total += n_mcts  # total number of environment steps (counts the mcts steps)
                if terminal:
                    break
                else:
                    mcts.forward(a, s1)

                # Finished episode
            episode_returns.append(R)  # store the total episode return
            timepoints.append(t_total)  # store the timestep count of the episode return

            if R > R_best:
                a_best = a_store
                seed_best = seed
                R_best = R
                print('new best with seed {} had the R {} and the moves were {}'.format(seed_best, R_best, a_best))
            print('Finished episode {}, total return: {}, total time: {} sec'.format(ep, np.round(R, 2),
                                                                                         np.round((time.time() - start),
                                                                                                  1)))


# def main(game,n_ep,n_mcts,max_ep_len,lr,c,gamma,data_size,batch_size,temp,n_hidden_layers,n_hidden_units, skip_frame):
#     preDefinedActions = []
#     for _ in range(100):
#         preDefinedActions.append(1)
#     env = gym.make("Pong-ram-v0")
#     env.reset()
#     terminal = False
#     env.seed(2566437)
#     for action in preDefinedActions:
#         s1, _,_,_ = env.step(action)
#         env.render("human")
#     env.reset()
#     env.seed(2566437)
#     for action in preDefinedActions:
#         s2, _,_,_ = env.step(action)
#
#     if(s1.all() == s2.all()):
#         print("hurray")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='Pong-v0', help='Training environment')
    parser.add_argument('--n_ep', type=int, default=50, help='Number of episodes')
    parser.add_argument('--n_mcts', type=int, default=50, help='Number of MCTS traces per step')  #
    parser.add_argument('--max_ep_len', type=int, default=10000, help='Maximum number of steps per episode')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--c', type=float, default=1.5, help='UCT constant')
    parser.add_argument('--temp', type=float, default=0.5,
                        help='Temperature in normalization of counts to policy target')
    parser.add_argument('--gamma', type=float, default=0.975, help='Discount parameter')  #
    parser.add_argument('--data_size', type=int, default=1000, help='Dataset size (FIFO)')
    parser.add_argument('--batch_size', type=int, default=32, help='Minibatch size')
    parser.add_argument('--window', type=int, default=25, help='Smoothing window for visualization')

    parser.add_argument('--n_hidden_layers', type=int, default=2, help='Number of hidden layers in NN')
    parser.add_argument('--n_hidden_units', type=int, default=128, help='Number of units per hidden layers in NN')
    parser.add_argument('--skip_frame', type=int, default=1,
                        help='Number of frames skipped between two agent observations')

    args = parser.parse_args()
    main(game=args.game,n_ep=args.n_ep,n_mcts=args.n_mcts,
                                        max_ep_len=args.max_ep_len,lr=args.lr,c=args.c,gamma=args.gamma,
                                        data_size=args.data_size,batch_size=args.batch_size,temp=args.temp,
                                        n_hidden_layers=args.n_hidden_layers,n_hidden_units=args.n_hidden_units,skip_frame=args.skip_frame)

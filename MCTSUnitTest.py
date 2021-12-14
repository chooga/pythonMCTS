import gym
import time

from main import Model, MCTS
from PIL import Image
import numpy as np

seed = 9280596
actions = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 1, 3, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 3, 2, 2, 1, 2, 1, 1, 3, 1, 2, 2, 2, 2, 2, 2, 1, 2, 3, 2, 2, 3, 2, 3, 2, 2, 1, 3, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 3, 2, 3, 3, 2, 2, 2, 2, 3, 3, 2, 2, 3, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 1, 2, 2, 3, 2, 2, 1, 2, 1, 1, 2, 2, 3, 2, 1, 2, 3, 2, 2, 2, 2, 2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 3, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 3, 2, 2, 3, 1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 3, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 3, 1, 1, 2, 2, 3, 2, 2, 3, 2, 2, 3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 3, 2, 2, 1, 1, 3, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 3, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 3, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 3, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 3, 2, 1, 3, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 3, 1, 1, 2, 2, 2, 3, 2, 1, 2, 2, 2, 2, 1, 1, 2, 1, 2, 1, 1, 3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 3, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 3, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 1, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2]

image = [192,   0,   0,   0, 110,  38,   0,   7,  71,   1,  60,  59,   0,   0,   0,  62, 255,   0, 255, 253,   0,  22,   0,  24, 128,  32,   1,  86, 247,  86, 247, 86, 247, 134, 243, 245, 243, 240, 240, 242, 242,  32,  32,  64,  64,  64, 188,  65, 189,   0,  22, 109,  37,  37,  60,   0,   0,   0,   0, 109, 109,  37,  37, 192, 192, 192, 192,   1, 192, 202, 247, 202, 247, 202, 247, 202, 247,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]


# img = Image.fromarray(data)
# img = img.resize((4,512))
# print(img.size)
# img.show()
env = gym.make('Pong-ramNoFrameskip-v4')
mctsEnv = gym.make('Pong-ramNoFrameskip-v4')
env = env.env
env.reset()
env.seed(seed)
# n_ones, n_twos, n_threes = 0, 0, 0
# for a in actions:
#     if(a == 1):
#         n_ones += 1
#     if(a == 2):
#         n_twos += 1
#     if(a==3):
#         n_threes += 1
#
# print("1: {}, 2: {}, 3: {}".format(n_ones,n_twos,n_threes))
t_total = 0  # total steps
R_best = -np.Inf

for i in range(600):
    start = time.time()
    s = env.reset()
    R = 0.0  # Total return counter
    a_store = []
    seed = np.random.randint(1e7)  # draw some Env seed
    env.seed(seed)
    mctsEnv.reset()
    mctsEnv.seed(seed)

    mcts = MCTS(root_index=s, root=None, model=model, na=model.action_dim,
                gamma=gamma)  # the object responsible for MCTS searches TODO #na=model.action_dim
    for t in range(max_ep_len):
        # MCTS step
        mcts.search(n_mcts=n_mcts, c=c, env=env, mcts_env=mctsEnv, skip_frame=skip_frame)  # perform a forward search
        state, pi, V = mcts.return_results(temp)  # extract the root output

        # pi_changed = applyNoise(pi)
        D.store((state, pi, V))
        # Make the true step
        a = np.random.choice(len(pi), p=pi)
        print(convAction(a))
        a_store.append(convAction(a))
        #                 s1, r, terminal, _ = env.step(a+1)
        env.render("human")
        # #                if (r > 0):
        # #                    input("waiting")
        #                 R += r
        for skfr in range(skip_frame):
            s1, r, terminal, _ = env.step(convAction(a))
            s1 = np.array(s1) / 255
            #                    if (r > 0):
            #                        input("waiting")
            if (r != 0):
                print(f"scored{r}")
            R += r
            if terminal:
                break
        t_total += n_mcts  # total number of environment steps (counts the mcts steps)
        if terminal:
            break
        else:
            mcts.forward(a, s1)

    for skfr in range(4):
        s1, r, terminal, _ = env.step(actions[i])
        env.render("human")
        time.sleep(0.01)
        if terminal:
            break

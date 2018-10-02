from ddpg import *
import gc
import math
gc.enable()

ENV_NAME = 'InvertedPendulum-v1'
EPISODES = 100000
TEST = 10

try:
    import hfo
except ImportError:
    print('Failed to import hfo. To install hfo, in the HFO directory'\
        ' run: \"pip install .\"')
    exit

ENV_NAME = 'InvertedPendulum-v1'
EPISODES = 100000
TEST = 10


def distance(a, b):
    c = [0,0]
    c[0] = a[0] - b[0]
    c[1] = a[1] - b[1]
    return math.hypot(c[0], c[1])

def CalReward(state, next_state, status):
    reward = 0
    if status == hfo.GOAL:
        reward += 5
    reward = reward + distance([state[0], state[1]], [state[3], state[4]]) - distance([next_state[0], next_state[1]], [next_state[3], next_state[4]])
    reward = reward + 3 * distance([state[3], state[4]], [1,0])
    reward = reward - 3 * distance([next_state[3], next_state[4]], [1,0])#[1,0] is the goal position
    
    return reward

def env_step(hfo_env, action):
    state = hfo_env.getState()
    maximum = np.max(action[:4])
    ##print("action:  ")
    ##print(action)
    a_c = np.where(action[:4] == maximum)
    ##print("raw a_c")
    ##print(a_c)
    a_c = a_c[0][0]
    ##print("a_c")
    ##print(a_c)
    if a_c == 0:
        hfo_env.act(hfo.DASH, action[4], action[5])
        ##print("dash")
    elif a_c == 1:
        hfo_env.act(hfo.TURN, action[6])
        ##print("turn")
    elif a_c == 3 and int(state[5]) == 1:
        hfo_env.act(hfo.KICK, action[8], action[9])
        ##print("kick")
    else:
        # hfo_env.act(hfo.TACKLE, action[7])
        ##print("tackle")
        hfo_env.act(hfo.DASH, action[4], action[5])
    status = hfo_env.step()
    next_state = hfo_env.getState()
    reward = CalReward(state, next_state, status)
    if status == hfo.IN_GAME:
        done = False
    else:
        done = True
    return next_state, reward, done, status



        

def main():
    hfo_env = hfo.HFOEnvironment()
    hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET)
    agent = DDPG()
    model_file=tf.train.latest_checkpoint('ckpt/')
    agent.saver.restore(agent.sess,model_file)
    for episode in range(EPISODES):
        status = hfo.IN_GAME
        
        while True:
            state = hfo_env.getState()
            # print(state)
            action = agent.noise_action(state)
            next_state, reward, done, status = env_step(hfo_env, action)

            agent.perceive(state,action,reward,next_state,done)
            if done:
                break
        if status == hfo.SERVER_DOWN:
            hfo_env.act(hfo.QUIT)
            exit()
        if episode % 100 == 0 and episode > 100:
            total_reward = 0
            for i in range(TEST):
                # state = env.reset()
                while True:
                    action = agent.action(state)
                    state, reward, done, status = env_step(hfo_env, action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            agent.saver.save(agent.sess,'ckpt/mnist.ckpt',global_step=episode)
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
        


if __name__ == '__main__':
    main()
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

def CalReward(agent, state, next_state, status):
    reward = 0
    # state[9] = 0
    # if min(state) == -2 or state[0] == -1 or state[1] == -1 or state[0] ==1 or state[1] == 1:
    #     state[9] = -2
    #     return -100
    if agent.has_kicked:
        if ~agent.laststep_haskicked:
            reward += 1
    if status == hfo.GOAL:
        reward += 5
    reward = reward + distance([state[0], state[1]], [state[3], state[4]]) - distance([next_state[0], next_state[1]], [next_state[3], next_state[4]])
    reward = reward + 3 * distance([state[3], state[4]], [1,0])
    reward = reward - 3 * distance([next_state[3], next_state[4]], [1,0])#[1,0] is the goal position
    state[9]=-2
    return reward

def state_violated(state):
    state_invalid = False
    state9 = state[9]
    state[9] = 0
    if min(state) == -2 or state[0] == -1 or state[1] == -1 or state[0] ==1 or state[1] == 1:
        state_invalid = True
        
    state[9] = state9
    return state_invalid

def env_step(agent, hfo_env, action):
    state = hfo_env.getState()
    
    if agent.has_kicked:
        agent.laststep_haskicked = True
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
        agent.has_kicked = True
        if_kick = True
        ##print("kick")
    else:
        # hfo_env.act(hfo.TACKLE, action[7])
        ##print("tackle")
        pass
    status = hfo_env.step()
    next_state = hfo_env.getState()
    fuck = state_violated(next_state)
    reward = CalReward(agent, state, next_state, status)
    
    if fuck:
        done = True
    elif status == hfo.IN_GAME:
        done = False
    else:
        done = True
    return next_state, reward, done, status



        

def main():
    hfo_env = hfo.HFOEnvironment()
    hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET)
    agent = DDPG()
    # model_file=tf.train.latest_checkpoint('ckpt/')
    # agent.saver.restore(agent.sess,model_file)
    for episode in range(130):
        status = hfo.IN_GAME
        stop_perceive = False
        with open('/home/ruizhao/Desktop/a.txt', 'a') as f:
            print('Hello World!', file=f)
        while True:
            
            state = hfo_env.getState()
            # print(state)
            action = agent.noise_action(state)
            print(action)
            next_state, reward, done, status = env_step(agent, hfo_env, action)
            # print(reward)
            if state_violated(next_state):
                # print("hhhhhhhhhhhhhh")
                stop_perceive = True
            if not stop_perceive:
                # print(state, next_state,done)
                # print(stop_perceive)
                agent.perceive(state,action,reward,next_state,done)
            if status != hfo.IN_GAME:
                break
        if status == hfo.SERVER_DOWN:
            hfo_env.act(hfo.QUIT)
            exit()
        # print(episode)
        # print(episode % 100 == 0 and episode > 100)
        if episode % 100 == 0 and episode > 100:
        # if True:
            total_reward = 0
            for i in range(TEST):
                # state = env.reset()
                while True:
                    state = hfo_env.getState()
                    action = agent.action(state)
                    next_state, reward, done, status = env_step(agent, hfo_env, action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            agent.saver.save(agent.sess,'ckpt/mnist.ckpt',global_step=episode)
            print('             episode: ', episode, 'Evaluation Average Reward:', ave_reward)
        


if __name__ == '__main__':
    main()
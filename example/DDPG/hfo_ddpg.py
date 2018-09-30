from ddpg import *
import gc
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


def CalReward(state, next_state):
    '''
    to be completed
    '''
    return 3

def env_step(hfo_env, action):
    state = hfo_env.getState()
    hfo_env.act(hfo.DASH, action[4], action[5])
    hfo_env.step()
    next_state = hfo_env.getState()
    reward = CalReward(state, next_state)
    if hfo_env.status == hfo.IN_GAME:
        done = False
    else:
        done = True
    return next_state, reward, done



        

def main():
    hfo_env = hfo.HFOEnvironment()
    hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET)
    agent = DDPG()
    '''
    DDPG need to change at the initial parameter
    '''
    for episode in range(EPISODES):
        status = hfo.IN_GAME
        count = 0
        
        while True:
            state = hfo_env.getState()
            action = agent.noise_action(state)
            next_state, reward, done = env_step(hfo_env, action)

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
                    state, reward, done = env_step(hfo_env, action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
        



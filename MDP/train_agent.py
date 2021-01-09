import sys
import time
from deep_q_network import DQNAgent
from double_deep_q_network import DoubleDQNAgent

from base_rl_agent_torch import ReplayMemory
from base_neural_model import EstimatorModelBase
from advantage_neural_model import AdvantageModel
from Data_binary import *

def setup_env(env_name, max_cost):
    if env_name == "survey":
        env = Data(unknown_rate=1)
        env.loadfile("survey.csv")
    elif env_name == "hcv":
        env = Data(unknown_rate=1)
        env.loadfile("hcv.csv") # change this to the test file
    elif env_name == "liver":
        env = Data(unknown_rate=1)
        env.loadfile("liver.csv") # change this to the test file
    else:
        print(env_name, "is not a valid environment name.")
        return None
              
    env.normalize()
    env.alpha = 0
    env.cluster_K_means(7)
    env.max_cost = max_cost
    env.split(0.80)
    return env
    
def train_and_save(env_name, max_cost, gamma=0.95, ep_decay=0.997, max_episodes=1000):
    start = time.time()
    env = setup_env(env_name, max_cost)
    if env == None:
        return
    
    MAX_EPISODES = max_episodes
    MAX_STEPS = 32
    BATCH_SIZE = 32
    buffer = ReplayMemory(10000)

    # Initiate the agent
    model = AdvantageModel
    agent = DoubleDQNAgent(env, 
                           model, 
                           buffer,
                           learning_rate=0.01,
                           max_steps=32, 
                           max_episodes=32,
                           gamma=gamma,
                           epsilon_decay=ep_decay,
                           exploration_penalty=0.,
                           verbose=0 # Verbosity level
                          )
    episode_rewards = agent.train(env, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)
    
    
    agent_string = "env"+env_name+"-max_cost"+str(max_cost)+"-gamma"+str(agent.gamma)+"-ep_decay"+str(agent.epsilon_decay)+"-max_episodes"+str(max_episodes)
    agent.save_model("saved_models/"+agent_string+".pt")
    agent.save_episode_rewards("metrics/" + agent_string+".pkl")
    print("Done. Took:", time.time()-start,"seconds")

if __name__ == '__main__':
    args = sys.argv
    # filename, max_cost, gamma, ep_decay, episodes
    train_and_save(args[1], float(args[2]), float(args[3]), float(args[4]), int(args[5]))
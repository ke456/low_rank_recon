from deep_q_network import DQNAgent
from double_deep_q_network import DoubleDQNAgent

from base_rl_agent_torch import ReplayMemory
from base_neural_model import EstimatorModelBase
from advantage_neural_model import AdvantageModel
import utils
import random
import copy
import time
from Data_binary import *

def testAgent(env, test_env, env_name, max_cost, gamma=0.95, max_eps=4000, epsilon_decay=0.9983):
    
    MAX_EPISODES = max_eps
    MAX_STEPS = 32
    BATCH_SIZE = 32
    buffer = ReplayMemory(10000)

    # Initiate the agent
    model = AdvantageModel
    agent = DoubleDQNAgent(env, 
                           model, 
                           buffer,
                           max_steps=MAX_STEPS, 
                           max_episodes=MAX_EPISODES,
                           gamma=gamma,
                           epsilon_decay=epsilon_decay,
                           exploration_penalty=-0.0,
                           verbose=0 # Verbosity level
                          )
    agent_string = "env"+env_name+"-max_cost"+str(max_cost)+"-gamma"+str(agent.gamma)+"-ep_decay"+str(agent.epsilon_decay)+"-max_episodes"+str(MAX_EPISODES)
    agent.load_model("saved_models/"+agent_string+".pt")
    agent.load_episode_rewards("metrics/" + agent_string+".pkl")
    episode_rewards = agent.episode_rewards
    
    # We don't want to take epsilon random actions
    agent.epsilon=0
    
    N=len(test_env.data)
    test_env.it = 0
    n = len(test_env.get(0)[1])-1
    k = 5
    
    distance = 0
    for it in range(N):
        observation = test_env.next_element()
        ob_cp = (copy.copy(observation[0]),copy.copy(observation[1]), copy.copy(observation[2]))
        ob_full = test_env.get(observation[0])
        ob_partial = np.array(ob_full[2][:n])
        
        done = False
        while not done:
            action = agent.get_action(observation,env)
            observation, reward, done, info = test_env.step(observation, action)
        ranks = test_env.compute_ranks(observation)
        K_most_similar_indices, K_most_similar_values, K_most_similar_data = env.K_most_similar(ranks, observation[2][:n], k)
        
        distance += np.sum(np.linalg.norm(ob_partial - K_most_similar_data, ord=2, axis=1))
        
    return distance

def testRandom(env, test_env, max_cost):
    
    N=len(test_env.data)
    test_env.it = 0
    n = len(test_env.get(0)[1])-1
    k = 5
    
    distance = 0
    for it in range(N):
        observation = test_env.next_element()
        ob_cp = (copy.copy(observation[0]),copy.copy(observation[1]), copy.copy(observation[2]))
        ob_full = test_env.get(observation[0])
        ob_partial = np.array(ob_full[2][:n])
        
        done = False
        cost = 0
        while not done:
            actions = test_env.actions(observation)
            r = -1
            if (len(actions) != 1):
                r = random.randint(0,len(actions)-2)
            action = actions[r]
            if action != -1 and action < len(observation[1])-1:
                cost += env.costs[action]
            observation, reward, done, info = test_env.step(observation, action)
            
        ranks = test_env.compute_ranks(observation)
        K_most_similar_indices2, K_most_similar_values2, K_most_similar_data2 = env.K_most_similar(ranks, observation[2][:n], k)
        
        distance += np.sum(np.linalg.norm(ob_partial - K_most_similar_data2, ord=2, axis=1))
        
    return distance

def runtest(env, test_env, max_eps=500, epsilon_decay=0.98):
    
    
    MAX_EPISODES = max_eps
    MAX_STEPS = 32
    BATCH_SIZE = 32
    buffer = ReplayMemory(10000)

    # Initiate the agent
    model = AdvantageModel
    agent = DoubleDQNAgent(env, 
                           model, 
                           buffer,
                           max_steps=MAX_STEPS, 
                           max_episodes=MAX_EPISODES,
                           learning_rate=0.01,
                           gamma=0.8,
                           epsilon_decay=epsilon_decay,
                           exploration_penalty=-0.0,
                           verbose=0 # Verbosity level
                          )
    episode_rewards = agent.train(env, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)
    
    # We don't want to take epsilon random actions
    agent.epsilon=0
    
    N=len(test_env.data)
    test_env.it = 0
    n = len(test_env.get(0)[1])-1
    k = 5
    
    dist1 = 0
    dist2 = 0
    dist3 = 0
    dist4 = 0
    for it in range(N):
        observation = test_env.next_element()
        ob_cp = (copy.copy(observation[0]),copy.copy(observation[1]), copy.copy(observation[2]))
        ob_full = test_env.get(observation[0])
        
        done = False
        while True:
                if done:
                    break;
                action = agent.get_action(observation,env)
                observation, reward, done, info = test_env.step(observation, action)
        ranks = test_env.compute_ranks(observation)
        ret1 = env.retrieve(ranks, observation[2][:n], k)
        ret3 = env.retrieve2(ranks, observation[2][:n], k)
        #print("r1:",ranks,sum([ np.linalg.norm(np.array(ob_full[2][:n]) - np.array(ret1[i][1]),2) for i in range(k)]))
        
        dist1 += sum([ np.linalg.norm(np.array(ob_full[2][:n]) - np.array(ret1[i][1]),2) for i in range(k)])
        dist3 += sum([ np.linalg.norm(np.array(ob_full[2][:n]) - np.array(ret3[i][1]),2) for i in range(k)])
        
        observation = ob_cp
        done = False
        cost = 0
        while True:
            if done:
                break;
            actions = test_env.actions(observation)
            r = -1
            if (len(actions) != 1):
                r = random.randint(0,len(actions)-2)
            action = actions[r]
            if action != -1 and action < len(observation[1])-1:
                cost += env.costs[action]
            observation, reward, done, info = test_env.step(observation, action)
            
        ranks = test_env.compute_ranks(observation)
        ret2 = env.retrieve(ranks, observation[2][:n], k)
        ret4 = env.retrieve2(ranks, observation[2][:n], k)
        #print("r2:",ranks,sum([ np.linalg.norm(np.array(ob_full[2][:n]) - np.array(ret2[i][1]),2) for i in range(k)]))
        #print("re:", test_env.compute_ranks(ob_full))
        #print()
        
        dist2 += sum([ np.linalg.norm(np.array(ob_full[2][:n]) - np.array(ret2[i][1]),2) for i in range(k)])
        dist4 += sum([ np.linalg.norm(np.array(ob_full[2][:n]) - np.array(ret4[i][1]),2) for i in range(k)])
        
    return dist1,dist2,dist3,dist4,episode_rewards
    
def plot_reward_per_episode(episode_rewards):
    import matplotlib.pyplot as plt
    episode_rewards = episode_rewards[0:]
    x = [i for i in range(len(episode_rewards)//100)]
    avg = [ sum(episode_rewards[i*100:(i+1)*100])/100 for i in range(len(episode_rewards)//100)]
    plt.plot(x, avg)

        
    
    
    
    
    
    
    
    
    
from runtest import *
from train_select import get_num_features

max_eps=500
epsilon_decay=0.98

#tests = ['parkinsons', 'cleaveland_heart']
tests = ['hcv']

def testAgent(env, test_env, env_name, max_cost, gamma=0.95, max_eps=4000, learning_rate=0.01, epsilon_decay=0.9983, update_steps=100):
    
    MAX_EPISODES = max_eps
    MAX_STEPS = 32
    BATCH_SIZE = 32
    buffer = ReplayMemory(10000)

    # Initiate the agent
    model = AdvantageModel
    agent = DoubleDQNAgent(env, 
                           model, 
                           buffer,
                           learning_rate=learning_rate,
                           max_steps=32, 
                           max_episodes=32,
                           gamma=gamma,
                           epsilon_decay=ep_decay,
                           exploration_penalty=0.,
                           update_steps=update_steps,
                           verbose=0 # Verbosity level
                          )
    agent_string = "env"+env_name+"-max_cost"+str(max_cost)+"-gamma"+str(agent.gamma)+"-ep_decay"+str(agent.epsilon_decay)+"-max_episodes"+str(MAX_EPISODES)+"-update_steps"+str(update_steps)
    agent.load_model("saved_models/"+agent_string+".pt")
    agent.load_episode_rewards("metrics/" + agent_string+".pkl")
    episode_rewards = agent.episode_rewards
    
    # We don't want to take epsilon random actions
    agent.epsilon=0
    
    N=len(test_env.data)
    test_env.it = 0
    n = len(test_env.get(0)[1])-1
    k = 5

    steps = [ [] for i in range(N) ]
    for it in range(N):
        observation = test_env.next_element()
        ob_cp = (copy.copy(observation[0]),copy.copy(observation[1]), copy.copy(observation[2]))
        ob_full = test_env.get(observation[0])
        ob_partial = np.array(ob_full[2][:n])
        
        done = False
        while not done:
            action = agent.get_action(observation, test_env)
            steps[it].append(action)
            observation, reward, done, info = test_env.step(observation, action)
    return steps

def write_steps(fname, steps):
    f = open(fname, 'w')
    for s in steps:
        for i in range(len(s)):
            f.write(str(s[i]))
            if i == len(s)-1:
                f.write("\n")
            else:
                f.write(",")
    f.close()
    
def read_params_from_file(test, fname):
    f = open(fname, 'r')
    lines = f.readlines()
    params = {}
    for line in lines:
        s = line.split(',')
        s = [float(s[0]),float(s[1]), float(s[2]), int(s[3]), float(s[4]), int(s[5])]

        params[s[0]] = s[1:]
    return params
            
    
# Test the agent
for test in tests:
    print("Running", test)
    N = 10
    budgets = [(i+1)/N for i in range(N)]
    costs = read_costs("csv_files/partitioned_data/" + test + "_cost.csv")
    
    env = Data(unknown_rate=1)
    env.loadfile_noshuffle("csv_files/partitioned_data/" + test + "_training.csv")
    env.cluster_K_means(7)
    
    test_env = Data(unknown_rate=1)
    test_env.loadfile_noshuffle("csv_files/partitioned_data/" + test + "_test.csv")
    test_env.cluster_K_means(7)
    
    env.set_costs(costs)
    test_env.set_costs(costs)
    
    params = read_params_from_file(test, "tuned_params/"+test+"_model_parameters.csv")
    for c in budgets:
        print("at budget:", c)
        gamma, ep_decay, eps, lr, update_steps = params[c] 
        print("Using parameters:", params[c])
        env.max_cost = c
        test_env.max_cost = c
        steps = testAgent(env,test_env, test, c, gamma=gamma, max_eps=eps, learning_rate=lr, epsilon_decay=ep_decay, update_steps=update_steps)
        write_steps("agent_runs/" + test + str(c*N) + ".csv",steps)
    print("Done", test)
    print("\n\n")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
from runtest import *

max_eps=500
epsilon_decay=0.98

tests = ['hcv', 'liver', 'survey']

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

    steps = [ [] for i in range(N) ]
    for it in range(N):
        observation = test_env.next_element()
        ob_cp = (copy.copy(observation[0]),copy.copy(observation[1]), copy.copy(observation[2]))
        ob_full = test_env.get(observation[0])
        ob_partial = np.array(ob_full[2][:n])
        
        done = False
        while not done:
            action = agent.get_action(observation,env)
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
            
for test in tests:
    print("Running", test)
    N = 10
    budgets = [(i+1)/N for i in range(N)]
    costs = read_costs("csv_files/partitioned_data/" + test + "_cost.csv")
    gammas = {}
    for cost in budgets:
        if cost >= 0.5:
            gammas[cost] = 0.8
        elif cost == 0.4:
            gammas[cost]=0.85
        else:
            gammas[cost]=0.7
    
    env = Data(unknown_rate=1)
    env.loadfile_noshuffle("csv_files/partitioned_data/" + test + "_training.csv")
    env.cluster_K_means(7)
    
    test_env = Data(unknown_rate=1)
    test_env.loadfile_noshuffle("csv_files/partitioned_data/" + test + "_test.csv")
    test_env.cluster_K_means(7)
    
    env.set_costs(costs)
    test_env.set_costs(costs)
    
    for c in budgets:
        print("at budget:", c)
        env.max_cost = c
        test_env.max_cost = c
        steps = testAgent(env,test_env, test, c, gamma=gammas[c], max_eps=500, epsilon_decay=0.98)
        write_steps("agent_runs/" + test + str(c*N) + ".csv",steps)
    print("Done", test)
    print("\n\n")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
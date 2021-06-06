from train_agent_naive import *
from runtest import testAgent
from itertools import product
import random

# Tuning this off will train the model with certain configs.
ENABLE_TUNING = True
# Turning this false will overwrite all the previous models.
OVERWRITE_ALL_PREVIOUS_MODELS = False

def train_select(test):
    if ENABLE_TUNING:
        print("Hyper-param tuning enabled for ", test)
        print(get_num_features(test))

        N = 10
        budgets = [(i+1)/N for i in range(N)] 

        tuning_params = ["gamma", "epsilon_decay/episodes/learning_rate", "target_update_steps"]
        # We tune over a cartesian product of the hyper-parameters below.
        candidate_gamma = [0.6, 0.7, 0.75, 0.8, 0.9]
        candidate_ep_decay = [(0.98, 500, 0.01), (0.99, 500, 0.01)]
        candidate_update_steps = [25, 50, 100]

        best_params = []
        for cost in budgets:
            results = {}
            print("cost:", cost)
            for gamma, (ep_decay, eps, lr), update_steps in product(candidate_gamma, candidate_ep_decay, candidate_update_steps):
                print("Training gamma", gamma, "ep_decay", ep_decay, "episodes", eps, "learning_rate", lr, "update_steps", update_steps)
                train_and_save(test, cost, gamma, ep_decay=ep_decay, max_episodes=eps, learning_rate=lr, update_steps=update_steps, overwrite=OVERWRITE_ALL_PREVIOUS_MODELS)
                print("Evaluating...", end='')
                score = eval_agent(test, cost, gamma, ep_decay, eps, lr, update_steps)
                print(" scored", score)
                results[gamma, (ep_decay, eps, lr), update_steps] = score
            print()
            gamma, (ep_decay, eps, lr), update_steps = min(results, key=results.get)
            print("Tuning for cost",cost,"has finished. The best set of hyperparameters were:")
            print("gamma", gamma, "ep_decay", ep_decay, "episodes", eps, "learning_rate", lr, "update_steps", update_steps)
            print("Which had scored:", results[gamma, (ep_decay, eps, lr), update_steps])
            print()
            best_params.append((cost, gamma, ep_decay, eps, lr, update_steps))

        save_hyperparams_to_file(test, best_params)

    else:
        print("Training on known best hyper-params for", test)
        N = 10
        budgets = [(i+1)/N for i in range(N)] # list of maximum budgets
        #costs = [0.4]
        gammas = {}
        for cost in budgets:
            if cost >= 0.5:
                gammas[cost] = 0.8
            elif cost == 0.4:
                gammas[cost]=0.85
            else:
                gammas[cost]=0.7


        for cost in budgets:
            print("At", cost*N)
            train_and_save(test, cost, gammas[cost],ep_decay=0.98, max_episodes=500)
    print("Finished", test)
    print("\n\n")
        
def save_hyperparams_to_file(test, best_params):
    fname = "tuned_params/"+test+"_model_parameters.csv"
    
    f = open(fname, 'w')
    for cost, gamma, ep_decay, eps, lr, update_steps in best_params:
        f.write(str(cost)+","+str(gamma)+","+str(ep_decay)+","+str(eps)+","+str(lr)+","+str(update_steps)+"\n")
    f.close()
        
def eval_agent(env_name, max_cost, gamma, ep_decay, eps, lr, update_steps):
    # Create validation set
    val_env = setup_val_env(env_name, max_cost)
    train_env = setup_env(env_name, max_cost)
    agent_distance = testAgent(train_env, val_env, env_name, max_cost, gamma=gamma, max_eps=eps, epsilon_decay=ep_decay, learning_rate=lr, update_steps=update_steps)
    #agent_distance = random.random()
    return agent_distance

def setup_val_env(env_name, max_cost):
    env = Data(unknown_rate=1)
    env.loadfile_noshuffle("csv_files/partitioned_data/"+env_name+"_val.csv")
    costs = read_costs("csv_files/partitioned_data/"+env_name+"_cost.csv")
              
    env.alpha = 0
    env.cluster_K_means(7)
    env.max_cost = max_cost
    env.set_costs(costs)
    return env
        
def get_num_features(test):
    env = Data(unknown_rate=1)
    env.loadfile("csv_files/" + test + ".csv") 
    return env.observation_space[0] - 1
        
if __name__ == '__main__':
    args = sys.argv
    train_select(args[1])
    
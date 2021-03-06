from Data_binary import *
import random

seed = 3
random.seed(seed)

# list of tests (just the root of the file name)
#tests = ['survey', 'liver', 'hcv', 'thyroid']
tests = ['parkinsons']
thyroid_costs = [0 for i in range(16)] + [22.78, 11.41, 14.51, 11.41, 14.51 + 11.41]
thyroid_costs = [cost/sum(thyroid_costs) for cost in thyroid_costs]
cleaveland_heart_costs = [1., 1., 1., 1., 7.27, 5.2, 15.5, 102.9, 87.3, 87.3, 87.3, 100.9, 102.9]
cleaveland_heart_costs = [cost/sum(cleaveland_heart_costs) for cost in cleaveland_heart_costs]

for test in tests:
    # first create the environment and load the data
    env = Data(unknown_rate=1)
    env.loadfile("csv_files/" + test + ".csv") 
    print(test)
    if test != 'thyroid':
        env.normalize()
    env.alpha = 0
    env.cluster_K_means(7)
    
    # find some random costs
    N = len(env.data[0][1])
    if test == 'thyroid':
        costs = thyroid_costs
    if test == 'cleaveland_heart':
        costs = cleaveland_heart_costs
    else:
        costs = [ random.randint(1,100) for i in range(N) ]
        costs = [costs[i] / sum(costs) for i in range(N)]
    
    env.set_costs(costs)
    
    # partition data and write to file
    test_env = env.split(0.80)
    
    if test in ['survey', 'thyroid']:
        validation_ratio=0.9
    else:
        validation_ratio=0.8
    val_env = test_env.split(validation_ratio)
    
    
    val_env.write_data("csv_files/partitioned_data/" + test + "_val.csv")
    env.write_data("csv_files/partitioned_data/" + test + "_training.csv")
    test_env.write_data("csv_files/partitioned_data/" + test + "_test.csv")
    env.write_cost("csv_files/partitioned_data/" + test + "_cost.csv")
    env.write_k_means("csv_files/partitioned_data/" + test + "_k_means.csv")
    
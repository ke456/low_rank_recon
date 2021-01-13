from Data_binary import *
import random

seed = 3
random.seed(seed)

# list of tests (just the root of the file name)
tests = ['survey', 'liver', 'hcv', 'thyroid']

for test in tests:
    # first create the environment and load the data
    test_env = Data(unknown_rate=1)
    test_env.loadfile_noshuffle("csv_files/partitioned_data/" + test + "_test.csv")
    print(test)
    test_env.alpha = 0
    test_env.cluster_K_means(7)
    
    # partition data and write to file
    val_env = test_env.split(0.90)
    test_env.write_data("csv_files/partitioned_data/" + test + "_test.csv")
    val_env.write_data("csv_files/partitioned_data/" + test + "_test.csv")
    
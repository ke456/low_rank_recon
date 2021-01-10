from train_agent import *

tests = ['hcv', 'liver', 'survey']

for test in tests:
    N = 10
    print("Starting to train for", test)
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
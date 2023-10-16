import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from folk_tables import protected_attributes, error_DP

def aposteriori(s_i, a_i, budget, n_repeat, random_seed=42):

    # 0 is SU, 1 is SS
    assert s_i in [0,1], "Strategy not supported"

    # create a np rng
    rng = np.random.default_rng(random_seed)

    print('================== Running e(strat^1)/e(strat^K) ==================')

    # Based on the fixed agent a_i
    attr_i = protected_attributes[a_i] 
    total_agents = len(protected_attributes)

    # load the dictionary storing individual strategies
    results = pickle.load(open(f'./results/results_{s_i}_{budget}.pkl', 'rb'))

    # load the ground truth demographic parity
    ground_truth_dp = pickle.load(open('./my_data/ground_truth_dp.pkl', 'rb'))

    # Repeat 5 times to get randomness
    total_results = []
    n_times = 5

    for _ in range(n_times):
        
        print(f'Running repetition {_+1}/{n_times}')

        joint_results = dict()
        for k in range(0, total_agents):  # Choose the number of agents to collaborate

            print(f'Running {k+1}/{total_agents} agents')

            # select the agents to collaborate except the a_i
            possible_agents = list(range(total_agents))
            possible_agents.remove(a_i)
            selected_agents = rng.choice(possible_agents, size=k, replace=False)
            
            joint_results[k] = [list(), list()]
            
            for r in range(n_repeat):
                X_i, y_i, e_i = results[attr_i][s_i][budget][r]
                X_tot = X_i
                y_tot = y_i

                for a_j in selected_agents:

                    attr_j = protected_attributes[a_j]
                    X_j, y_j, _ = results[attr_j][s_i][budget][r]
                    
                    X_tot = pd.concat([X_tot, X_j], ignore_index=True)
                    y_tot = pd.concat([y_tot, y_j], ignore_index=True)

                e_aposteriori = error_DP(X_tot, y_tot, attr_i, ground_truth_dp)
                joint_results[k][0].append(e_i)
                joint_results[k][1].append(e_aposteriori)
                
            joint_results[k] = np.mean(joint_results[k][0])/np.mean(joint_results[k][1])
            # print(f'Average error {k}: {joint_results[k]}')

        total_results.append(list(joint_results.values()))
    

    # plot the mean and std
    avg_results = np.mean(total_results, axis=0)
    std_results = np.std(total_results, axis=0)
    plt.plot(range(1,total_agents+1), avg_results, marker='o', color='blue')
    plt.fill_between(range(1,total_agents+1), avg_results-std_results, avg_results+std_results, color='blue', alpha=0.2)
    plt.xlabel(r'Number of agents $(K)$')
    if s_i == 0:
        plt.ylabel(r'$e(SU^1)/e(SU^K)$')
    else:
        plt.ylabel(r'$e(SS^1)/e(SS^K)$')
    plt.savefig(f'results/plot_thm1_strat{s_i}_agent{a_i}_seed{random_seed}_budget{budget}.png')

def aposteriori_forall(s_i, budget, n_repeat, random_seed=42):
    # 0 is SU, 1 is SS
    assert s_i in [0,1], "Strategy not supported"

    # create a np rng
    rng = np.random.default_rng(random_seed)

    print('================== Running e(strat^1)/e(strat^K) ==================')

    total_agents = len(protected_attributes)

    # load the dictionary storing individual strategies
    results = pickle.load(open(f'./results/results_{s_i}_{budget}.pkl', 'rb'))

    # load the ground truth demographic parity
    ground_truth_dp = pickle.load(open('./my_data/ground_truth_dp.pkl', 'rb'))

    # Repeat 5 times to get randomness
    n_times = 20

    agentwise_mean = []
    agentwise_std = []

    for a_i in range(total_agents):
        attr_i = protected_attributes[a_i] 
        print(f'Running agent {attr_i} {a_i+1}/{total_agents}')
        
        total_results = []

        for _ in range(n_times):
            
            print(f'Running repetition {_+1}/{n_times}')

            joint_results = dict()
            for k in range(0, total_agents):  # Choose the number of agents to collaborate

                print(f'Running {k+1}/{total_agents} agents')

                # select the agents to collaborate except the a_i
                possible_agents = list(range(total_agents))
                possible_agents.remove(a_i)
                selected_agents = rng.choice(possible_agents, size=k, replace=False)
                
                joint_results[k] = [list(), list()]
                
                for r in range(n_repeat):
                    X_i, y_i, e_i = results[attr_i][s_i][budget][r]
                    X_tot = X_i
                    y_tot = y_i

                    for a_j in selected_agents:

                        attr_j = protected_attributes[a_j]
                        X_j, y_j, _ = results[attr_j][s_i][budget][r]
                        
                        X_tot = pd.concat([X_tot, X_j], ignore_index=True)
                        y_tot = pd.concat([y_tot, y_j], ignore_index=True)

                    e_aposteriori = error_DP(X_tot, y_tot, attr_i, ground_truth_dp)
                    joint_results[k][0].append(e_i)
                    joint_results[k][1].append(e_aposteriori)
                    
                joint_results[k] = np.mean(joint_results[k][0])/np.mean(joint_results[k][1])

            total_results.append(list(joint_results.values()))

        agentwise_mean.append(np.mean(total_results, axis=0))
        agentwise_std.append(np.std(total_results, axis=0))
    
    agentwise_mean = np.array(agentwise_mean)
    agentwise_std = np.array(agentwise_std)

    # create a 5 x 2 subplot for 10 agents
    _, axs = plt.subplots(5, 2, figsize=(10, 20))
    axs = axs.flatten()
    for i in range(total_agents):
        axs[i].plot(range(1,total_agents+1), agentwise_mean[i], marker='o', color='blue')
        axs[i].fill_between(range(1,total_agents+1), agentwise_mean[i]-agentwise_std[i], agentwise_mean[i]+agentwise_std[i], color='blue', alpha=0.2)
        axs[i].set_xlabel(r'Number of agents $(K)$')
        if s_i == 0:
            axs[i].set_ylabel(r'$e(SU^1)/e(SU^K)$')
        else:
            axs[i].set_ylabel(r'$e(SS^1)/e(SS^K)$')
        axs[i].set_title(f'Agent {i} - {protected_attributes[i]}')
    plt.tight_layout()
    plt.savefig(f'results/plot_thm1_strat{s_i}_all_seed{random_seed}_budget{budget}.png')

def evolution_with_budget(s_i, a_i, n_repeat):
    # Based on the fixed agent a_i
    attr_i = protected_attributes[a_i] 

    # load the dictionary storing individual strategies
    results = pickle.load(open(f'./results/results_{s_i}_many.pkl', 'rb'))

    budgets = results[attr_i][s_i].keys()

    budgetwise_error_mean = []
    budgetwise_error_std = []
    
    for b in budgets:
        
        errors = []
        for r in range(n_repeat):
            
            _, _, e_i = results[attr_i][s_i][b][r]
            errors.append(e_i)
        
        budgetwise_error_mean.append(np.mean(errors))
        budgetwise_error_std.append(np.std(errors))
    
    budgetwise_error_mean = np.array(budgetwise_error_mean)
    budgetwise_error_std = np.array(budgetwise_error_std)

    plt.plot(budgets, budgetwise_error_mean, marker='o', color='blue')
    plt.fill_between(budgets, budgetwise_error_mean-budgetwise_error_std, budgetwise_error_mean+budgetwise_error_std, color='blue', alpha=0.2)
    plt.xlabel('Audit budget')
    plt.ylabel('Error')
    strategy_name = 'SU' if s_i == 0 else 'SS'
    plt.title(f'Agent {a_i} - {attr_i} | Strategy {strategy_name}')
    plt.savefig(f'results/plot_evb_strat{s_i}_agent{a_i}.png')

def evolution_with_budgets_forall(s_i):

    # load the dictionary storing individual strategies
    results = pickle.load(open(f'./results/results_{s_i}_many_0.pkl', 'rb'))

    budgets = list(results['MAR'][s_i].keys())
    total_agents = len(protected_attributes)
    n_repeat = len(results['MAR'][s_i][budgets[0]])

    print(f'Number of budgets: {len(budgets)}, number of agents: {total_agents}, number of repetitions: {n_repeat}')

    agent_error_mean = []
    agent_error_std = []

    for a_i in range(total_agents):
        attr_i = protected_attributes[a_i]
        budgetwise_error_mean = []
        budgetwise_error_std = []
        
        for b in budgets:
            
            errors = []
            for r in range(n_repeat):
                
                _, _, e_i = results[attr_i][s_i][b][r]
                errors.append(e_i)
            
            budgetwise_error_mean.append(np.mean(errors))
            budgetwise_error_std.append(np.std(errors))
        
        budgetwise_error_mean = np.array(budgetwise_error_mean)
        budgetwise_error_std = np.array(budgetwise_error_std)

        agent_error_mean.append(budgetwise_error_mean)
        agent_error_std.append(budgetwise_error_std)

    # create a 5 x 2 subplot for 10 agents
    _, axs = plt.subplots(5, 2, figsize=(10, 20))
    axs = axs.flatten()
    for i in range(total_agents):
        axs[i].plot(budgets, agent_error_mean[i], marker='o', color='blue')
        axs[i].fill_between(budgets, agent_error_mean[i]-agent_error_std[i], agent_error_mean[i]+agent_error_std[i], color='blue', alpha=0.2)
        axs[i].set_xlabel('Audit budget')
        axs[i].set_ylabel('Error')
        strategy_name = 'SU' if s_i == 0 else 'SS'
        axs[i].set_title(f'Agent {i} - {protected_attributes[i]} | Strategy {strategy_name}')
    plt.tight_layout()
    plt.savefig(f'results/plot_evb_strat{s_i}_all_0.png')


if __name__ == '__main__':
    

    # strategies = [(0, SU)]
    
    # SU collaboration gains
    # s_i = 0
    # a_i = 8
    # budget = 100
    # n_repeat = 5
    # random_seed = 100
    # aposteriori(s_i, a_i, budget, n_repeat, random_seed=random_seed)

    # SS collaboration gains for all
    s_i = 1
    budget = 100
    n_repeat = 5
    random_seed = 100
    aposteriori_forall(s_i, budget, n_repeat, random_seed=random_seed)

    # Evolution with budget
    # s_i = 1
    # a_i = 5
    # n_repeat = 5
    # evolution_with_budget(s_i, a_i, n_repeat)

    # Evolution with budget for all agents
    # s_i = 1
    # evolution_with_budgets_forall(s_i)

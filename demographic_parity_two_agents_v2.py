from sympy import *
import numpy as np
import numpy.ma as ma
import nashpy as nash
import operator

precision = 0.01 # Just for some assertions.
def assert_array_less_equal(x, y, err_msg='', verbose=True):
    from numpy.testing import assert_array_compare
    __tracebackhide__ = True  # Hide traceback for py.test
    assert_array_compare(operator.__le__, x, y, err_msg=err_msg,
                         verbose=verbose,
                         header='Some probabilities are not between 0 and 1.')


exp_id = np.random.randint(100000)
print("exp_id:", exp_id)
seed = 42
np.random.seed(seed)

'''
Set-up and probabilities
'''
# We assume that C0 and C1 are independant

Nrepet= 100 ## number of repetitions per sample level
proof_scheduler = True # to know if the last experiment on Nash equilibirum must be done
particular_budget = 11 # For which budget equilibirums are calculated
print_wng = True
B = [r*10 for r in range(1, 11)] + [r*100 for r in range(1, 11)] + [r*1000 for r in range(1, 6)] # Budget for each agents
assert particular_budget < len(B), "the asked budget doesn't exist."

pC0=.7 ## P(C0 = 1)
pYC01=.6 ## P(Y = 1 | C0 = 1)
DP0=.17 ## Demographic parity on C0: P(Y = 1 | C0 = 1) - P(Y = 1 | C0 = 0)
pYC00 = pYC01 - DP0 ## P(Y = 1 | C0 = 0)
pY = pYC01*pC0 + pYC00*(1-pC0)

pC1=.1 ## P(C1 = 1)
DP1=.4 ## Demographic parity on C1: P(Y = 1 | C1 = 1) - P(Y = 1 | C1 = 0)
pYC11 = pY + DP1*(1 - pC1) ## P(Y = 1 | C1 = 1)

pYC10 = pYC11 - DP1 ## P(Y = 1 | C1 = 0)

all_p = [pC0, pYC01, DP0, pYC00, pC1, pYC11, DP1, pYC10]
assert_array_less_equal(np.zeros(8), all_p)
assert_array_less_equal(all_p, np.ones(8))


pYC00C10 = pYC00*pYC10/pY # P(Y = 1 | C0 = 0, C1 = 0) = P(Y = 1 | C0 = 0). P(Y = 1| C1 = 0) / P(Y = 1)
pYC01C10 = pYC01*pYC10/pY # P(Y = 1 | C0 = 1, C1 = 0) = P(Y = 1 | C0 = 1). P(Y = 1| C1 = 0) / P(Y = 1)
pYC00C11 = pYC00*pYC11/pY # P(Y = 1 | C0 = 0, C1 = 1) = P(Y = 1 | C0 = 0). P(Y = 1| C1 = 1) / P(Y = 1)
pYC01C11 = pYC01*pYC11/pY # P(Y = 1 | C0 = 1, C1 = 1) = P(Y = 1 | C0 = 1). P(Y = 1| C1 = 1) / P(Y = 1)

assert abs(pYC00C10*(1-pC0)*(1-pC1) + pYC01C10*pC0*(1-pC1) + pYC00C11*(1-pC0)*pC1 + pYC01C11*pC0*pC1 - pY) < precision

'''
Main functions of the simulation
'''
def demographic_parity(samples, C):
    '''
    P(Y = 1 | C = 1) - P(Y = 1 | C = 0)
    Y is the target, C is the protected class
    '''
    if C == 'C0':
        i = 0
    else:
        i = 1
    sum_samples = np.sum(samples, axis = -1)
    n = len(samples[0])
    if not (0 < sum_samples[i] < n): # if the auditor doesn't test all subpopulations on C, we set that the demographic parity is null
        return 0
    else:
        return np.sum(ma.masked_array(samples[2], mask=1-samples[i]))/sum_samples[i] - np.sum(ma.masked_array(samples[2], mask=samples[i]))/(n-sum_samples[i])


def error_DP(samples, C):
    '''
    To calculate the margin of error between the empirical demographic parity calculated on the protected class C
    on the examples from 'samples'
    '''
    if C == 'C0':
        return np.abs(DP0 - demographic_parity(samples, C))
    else:
        return np.abs(DP1 - demographic_parity(samples, C))


def BlackBox(t0, t1):
    '''
    The black-box algorithm to audit
    Inputs: (x_0, x'_0, .... x"_0), (x_1, x'_1, .... x"_1)
    Where x=(x_0, x_1) \in X = (x, x', ..., x") are the queries.
    Outputs: (x_0, x'_0, .... x"_0), (x_1, x'_1, .... x"_1), (y, y', ..., y")
    '''
    assert abs(len(t0) - len(t1)) < precision
    n = len(t0)
    # random numbers to generate probabilities
    t2 = np.random.random(n)
    # the output vector
    t3 = t0*((t2 < pYC01C10)*(1-t1)+ (t2 < pYC01C11)*t1) + (1-t0)*((t2 < pYC00C10)*(1-t1)+ (t2 < pYC00C11)*t1)
    return np.vstack((t0, t1, t3)) # stacking inputs-outputs together

def RS(n, C = None):
    '''
    Random Sampling.
    C is not used in the method.
    '''
    t0 = np.random.random(n) < pC0 # t0[i] = 1_{i \in C_0}
    t1 = np.random.random(n) < pC1 # t1[i] = 1_{i \in C_1}

    return BlackBox(t0, t1)


def SSNC(n, C, print_wng = False):
    '''
    Non-Collaborative Stratified Sampling.
    '''
    if n % 2 != 0 and print_wng:
        print("WARNING: uneven n, one mising query")
    sub_n = n//2
    t0 = np.concatenate([np.zeros(sub_n), np.ones(sub_n)]) # sub_n \notin C, sub_n \in C
    if C == 'C0':
        t1 = np.random.random(sub_n*2) < pC1 # t1[i] = 1_{i \in C_1}
        return BlackBox(t0, t1)
    else:
        t1 = np.random.random(sub_n*2) < pC0 # t0[i] = 1_{i \in C_0}
        return BlackBox(t1, t0)

def SSC(n, C = None, print_wng = False):
    '''
    Collaborative Stratified Sampling.
    C is not used in the method.
    '''
    if n % 4 != 0 and print_wng:
        print("WARNING: n is not divisible by 4, mising query(ies)")
    sub_n = n//4
    # sub_n \in \C0\C1, sub_n \in C1\C0, sub_n \in C0\C1, sub_n \in C0C1
    t0 = np.concatenate([np.zeros(sub_n), np.zeros(sub_n),  np.ones(sub_n), np.ones(sub_n)])
    t1 = np.concatenate([np.zeros(sub_n), np.ones(sub_n), np.zeros(sub_n), np.ones(sub_n)])
    return BlackBox(t0, t1)

def OSNC(n, C):
    '''
    Non-Collaborative Optimal Sampling.
    '''
    if C == 'C0':
        # P0, P1 = pC0, 1 - pC0 -> wrong !
        P0, P1 = pYC01, pYC00
    else:
        # P0, P1 = pC1, 1 - pC1 -> wrong !
        P0, P1 = pYC11, pYC10
    # Finding the optimal strategy by solving the Equation (?) using sympy. Is it always equivalent to proportionate stratified sampling ??
    p1, p0, m, a = symbols("p1 p0 n a")
    espilon = 2*sqrt((p1*(1-p1))/(a*m)) + 2*sqrt((p0*(1-p0))/((1-a)*m))
    evaluation = espilon.subs(m, n)
    evaluation = evaluation.subs(p0, P0)
    evaluation = evaluation.subs(p1, P1)
    derivative = evaluation.diff(a)
    s = solve(derivative, a)
    if s != []:
        alpha = int(s[0]*n)
        t0 = np.concatenate([np.zeros(alpha), np.ones(n-alpha)])
        if C == 'C0':
            t1 = np.random.random(n) < pC1 # t1[i] = 1_{i \in C_1}
            return BlackBox(t0, t1)
        else:
            t1 = np.random.random(n) < pC0 # t0[i] = 1_{i \in C_0}
            return BlackBox(t1, t0)

def OSC(n, C):
    '''
    Collaborative Optimal Sampling.
    '''
    if C == 'C0':
        D = 'C1'
        # P0, P1 = pC0, 1 - pC0 -> wrong !
        # P2, P3 = pC1, 1 - pC1 -> wrong !
        P0, P1 = pYC01, pYC00
        P2, P3 = pYC11, pYC10
    else:
        D = 'C0'
        # P0, P1 = pC1, 1 - pC1 -> wrong !
        # P2, P3 = pC0, 1 - pC0 -> wrong !
        P0, P1 = pYC11, pYC10
        P2, P3 = pYC01, pYC00
    # Finding the optimal strategy by solving the Equation (?) using sympy.
    p1, p0, m, a = symbols("p1 p0 n a")
    espilon = 2*sqrt((p1*(1-p1))/(a*m)) + 2*sqrt((p0*(1-p0))/((1-a)*m))
    evaluation = espilon.subs(m, n)
    evaluation = evaluation.subs(p0, P0)
    evaluation = evaluation.subs(p1, P1)
    derivative = evaluation.diff(a)
    s = solve(derivative, a)
    if s != []:
        alpha = int(s[0]*n) # WRONG !
        t0 = np.concatenate([np.zeros(alpha), np.ones(n-alpha)])

        # How to sample considering the other class ?
        p1, p0, m, a = symbols("p1 p0 n a")
        espilon = 2*sqrt((p1*(1-p1))/(a*m)) + 2*sqrt((p0*(1-p0))/((1-a)*m))
        evaluation = espilon.subs(m, int(alpha*n))
        evaluation = evaluation.subs(p0, P2)
        evaluation = evaluation.subs(p1, P3)
        derivative = evaluation.diff(a)
        s = solve(derivative, a)
        if s != []:
            beta = float(s[0])
            t1 = np.concatenate([np.zeros(int(beta*alpha)), np.ones(alpha-int(beta*alpha)), np.zeros(int(beta*(n-alpha))), np.ones((n-alpha)-int(beta*(n-alpha)))])

            if C == 'C0':
                return BlackBox(t0, t1)
            else:
                return BlackBox(t1, t0)

'''
Creation of the queries and audit thanks to all strategies
'''
strategies = [[0, RS], [1, SSNC], [2, SSC], [3, OSNC], [4, OSC]]



res0 = [ [ [] for _ in B] for _ in strategies]
res1 = [ [ [] for _ in B] for _ in strategies]

for i, strat in strategies:
    for b in range(len(B)):
        budget = int(B[b])
        cur_res0 = []
        cur_res1 = []
        for _ in range(Nrepet):
            s0 = strat(budget, 'C0')
            s1 = strat(budget, 'C1')
            cur_res0.append([s0, error_DP(s0, 'C0')])
            cur_res1.append([s1, error_DP(s1, 'C1')])

        res0[i][b] = cur_res0
        res1[i][b] = cur_res1

res = [ [ [ [] for _ in B] for _ in strategies] for _ in strategies]
for i, strati in strategies:
    for j, stratj in strategies:
        for b in range(len(B)):
            budget = int(B[b])
            for r in range(Nrepet):
                s0, e0 = res0[i][b][r]
                s1, e1 = res1[j][b][r]
                s_tot = np.hstack((s0, s1))
                res[i][j][b].append([e0, e1, error_DP(s_tot, 'C0'), error_DP(s_tot, 'C1')])
# print(np.array(res).shape) # strat0, strat1, budget, repetition, measure

'''
Results
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.gcf().clear()
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
res= np.array(res)

m = len(strategies)

fig, axs = plt.subplots(m+1, m+1, sharex=True, sharey=True, figsize=(20, 20))
labels = [r'$\epsilon(DP_0)$ w/o $A_1$', r'$\epsilon(DP_1)$ w/o $A_0$', r'$\epsilon(DP_0)$ w $A_1$', r'$\epsilon(DP_1)$ w $A_0$']
strat_names = ['no sampling']+[ s.__name__ for _, s in strategies]
lines = ['--', '-']
colors = ['red', 'blue']
alphas = [0.3, 1]

# For Nash's study
MoE0 = np.zeros((m,m))
MoE1 = np.zeros((m,m))

for i in range(m):
    axs[i+1][0].set_ylabel(strat_names[i+1])
    axs[-1][i+1].set_xlabel(strat_names[i+1])

    # Without sharing data
    k = 0
    to_plot0 = np.mean(res[i][i].T, axis = 1)
    axs[i+1][0].plot(B, to_plot0[k], linestyle = lines[k//2], color = colors[k%2], label=labels[k])

    k = 1
    to_plot1 = np.mean(res[i][i].T, axis = 1)
    axs[0][i+1].plot(B, to_plot1[k], linestyle = lines[k//2], color = colors[k%2], label=labels[k])


    # Sharing data for each (strat, start')
    for j in range(m):
        to_plot = np.mean(res[i][j].T, axis = 1)
        # For Nash's study
        MoE0[i,j] = to_plot[2][particular_budget]
        MoE1[i,j] = to_plot[3][particular_budget]
        for k in range(4):
            axs[i+1][j+1].plot(B, to_plot[k], linestyle = lines[k//2], alpha = alphas[k//2], color = colors[k%2], label=labels[k])

axs[0][0].set_ylabel(strat_names[0])
axs[-1][0].set_xlabel(strat_names[0])


handles, labels = axs[-1][-1].get_legend_handles_labels()
lgd = axs[-1][2].legend(handles, labels, loc='center', bbox_to_anchor=(0.2,-0.4),  ncol=5)
plt.xscale('log')
plt.savefig('data/'+str(exp_id)+'_fig.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

print("Theoretical demographic parity on C_0:", DP0, "and on C_1:", DP1)


if proof_scheduler:
    print("Pareto study...")
    rewards0 = np.ones_like(MoE0) - MoE0
    rewards1 = np.ones_like(MoE1) - MoE1

    # Creation of the game
    common_audit = nash.Game(rewards0, rewards1)

    # Nash equilibriums
    equilibria = common_audit.support_enumeration()
    sols = []
    for e in equilibria:
        sols.append(e)
    if sols == []:
        print("There is no nash equilibirum !")
    else:
        for eq01 in sols:
            is_PO = True
            rewards_eq0, rewards_eq1 = common_audit[eq01]
            for eq_bis in equilibria:
                rewards_bis0, rewards_bis1 = common_audit[eq_bis]
                if (rewards_bis0 > rewards_eq0 and rewards_bis1>= rewards_eq1) or (rewards_bis0 >= rewards_eq0 and rewards_bis1> rewards_eq1):
                    is_PO = False
                    break
            eq0, eq1 = eq01
            if is_PO:
                if print_wng:
                    print("Pareto-efficient solution:")
                    s0 = np.argmax(eq0)
                    s1 = np.argmax(eq1)
                    print(" Strategy for agent 0 :", strat_names[s0+1])
                    print(" Strategy for agent 0 :", strat_names[s1+1])
                    print(" MoE for agent 0 :", np.round(MoE0[s0][s1],3), "sufficient ?", np.abs(DP0 - 0.2) < MoE0[s0][s1])
                    print(" MoE for agent 1 :", np.round(MoE1[s0][s1],3), "sufficient ?", np.abs(DP1 - 0.2) < MoE1[s0][s1])
                    print()
            else:
                if print_wng:
                    print("NOT Pareto-efficient solution:")
                    s0 = np.argmax(eq0)
                    s1 = np.argmax(eq1)
                    print(" Strategy for agent 0 :", strat_names[s0+1])
                    print(" Strategy for agent 0 :", strat_names[s1+1])
                    print(" MoE for agent 0 :", np.round(MoE0[s0][s1],3), "sufficient ?", np.abs(DP0 - 0.2) < MoE0[s0][s1])
                    print(" MoE for agent 1 :", np.round(MoE1[s0][s1],3), "sufficient ?", np.abs(DP1 - 0.2) < MoE1[s0][s1])
                    print()
                print("the scheduler is necessary.")

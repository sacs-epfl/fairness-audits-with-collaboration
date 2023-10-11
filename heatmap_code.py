# joint_results = dict()
# # initialize joint results
# for i, strat in strategies:
#     joint_results[i] = dict()
#     for j in range(len(protected_attributes)):
#         for k in range(j, len(protected_attributes)):
#             attr1 = protected_attributes[j]
#             attr2 = protected_attributes[k]
#             joint_results[i][(attr1, attr2)] = {b:[] for b in budgets}
#             joint_results[i][(attr2, attr1)] = {b:[] for b in budgets}

# # evaluate results for the joint strategy
# for i, strat in strategies:
#     print(f'Running strategy {strat.__name__} {i+1}/{len(strategies)}')

#     for j in range(len(protected_attributes)):
#         for k in range(j, len(protected_attributes)):
            
#             print(f'Running attributes {j} and {k}')

#             attr1 = protected_attributes[j]
#             attr2 = protected_attributes[k]

#             for b in budgets:
#                 print(f'Running budget {b}')
            
#                 for r in range(Nrepeat):
#                     # print(f'Running repetition {r+1}/{Nrepet}')
#                     X1, y1, e1 = results[attr1][i][b][r]
#                     X2, y2, e2 = results[attr2][i][b][r]
                    
#                     X_tot = pd.concat([X1, X2], ignore_index=True)
#                     y_tot = pd.concat([y1, y2], ignore_index=True)
                    
#                     e1_aposteriori = error_DP(X_tot, y_tot, attr1, ground_truth_dp)
#                     e2_aposteriori = error_DP(X_tot, y_tot, attr2, ground_truth_dp)

#                     joint_results[i][(attr1, attr2)][b].append(e1/e1_aposteriori)
#                     joint_results[i][(attr2, attr1)][b].append(e2/e2_aposteriori)



# # plot the results

# # Create an empty 2D array to store the error values
# num_attributes = len(protected_attributes)
# error_matrix = np.zeros((num_attributes, num_attributes))
# budget = budgets[-1]

# # Fill the error_matrix with error values from your joint_results dictionary
# for i, strat in strategies:
#     for j in range(num_attributes):
#         for k in range(num_attributes):
#             attr1 = protected_attributes[j]
#             attr2 = protected_attributes[k]
#             # Calculate the average error for the combination (attr1, attr2)
#             avg_error = np.mean(joint_results[i][(attr1, attr2)][budget])
#             error_matrix[j, k] = avg_error
        
# # Create a figure and axis
# fig, ax = plt.subplots()

# # Create the heatmap
# cax = ax.imshow(error_matrix, cmap='viridis', interpolation='nearest')

# # Add a colorbar
# cbar = fig.colorbar(cax)

# # Set axis labels and title
# ax.set_xticks(np.arange(num_attributes))
# ax.set_yticks(np.arange(num_attributes))
# ax.set_xticklabels(protected_attributes)
# ax.set_yticklabels(protected_attributes)
# ax.set_xlabel('Attribute 1')
# ax.set_ylabel('Attribute 2')
# ax.set_title('Error for Attribute Combinations')

# plt.xticks(np.arange(num_attributes), protected_attributes, rotation=45, ha='right')

# # Show the plot
# plt.savefig(f'results/heatmap_{random_seed}_{budgets[0]}.png')
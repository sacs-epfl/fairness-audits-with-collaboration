import os

import pandas as pd
import logging

def write_results(args, results):
    if not os.path.exists("results"):
        os.mkdir("results")

    agents: int = len(args.attributes_to_audit)

    results_file_name = "%s_%s_n%d_b%d" % (args.collaboration, args.sample, agents, args.budget)
    if args.unbias_mean:
        results_file_name += "_unbias"
    results_file_name += ".csv"
    results_file_path = os.path.join("results", results_file_name)

    logging.info("Writing results to %s", results_file_path)
    with open(results_file_path, "w") as results_file:
        results_file.write("dataset,collaboration,sample,agents,seed,budget,agent,attribute,dp_error\n")

        group_name = args.collaboration
        if args.collaboration == "apriori" and args.unbias_mean:
            group_name = "apriori (unbiased)"

        for result in results:
            results_file.write("%s,%s,%s,%d,%d,%d,%d,%s,%f\n" % (args.dataset, group_name, args.sample, agents, *result))


def merge_csv_files(filenames, output_filename):
    # List to store DataFrames
    dfs = []

    # Read each file into a DataFrame and append to the list
    for file in filenames:
        df = pd.read_csv(file)
        dfs.append(df)

    # Concatenate all DataFrames in the list
    merged_df = pd.concat(dfs, ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_filename, index=False)

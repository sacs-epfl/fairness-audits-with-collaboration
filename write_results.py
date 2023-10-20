import os


def write_results(args, results):
    if not os.path.exists("results"):
        os.mkdir("results")

    results_file_name = "%s_%s_n%d" % (args.collaboration, args.sample, args.agents)
    if args.unbias_mean:
        results_file_name += "_unbias"
    results_file_name += ".csv"
    results_file_path = os.path.join("results", results_file_name)
    with open(results_file_path, "w") as results_file:
        results_file.write("dataset,collaboration,sample,agents,seed,budget,agent,dp_error\n")

        group_name = args.collaboration
        if args.collaboration == "apriori" and args.unbias_mean:
            group_name = "apriori (unbiased)"

        for result in results:
            results_file.write("%s,%s,%s,%d,%d,%d,%d,%f\n" % (args.dataset, group_name, args.sample, args.agents, *result))

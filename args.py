import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default="stratified", choices=["uniform", "stratified"])
    parser.add_argument("--collaboration", type=str, default="none", choices=["none", "aposteriori", "apriori"])
    parser.add_argument("--collaborating-agents", type=str, default=None, help="The agents that collaborate, must match the number passed to the program with --agents. Must be passed in the form 1,3 (if agents 1 and 3 collaborate).")
    parser.add_argument("--agent-to-attribute", type=str, default=None, help="The attribute that each agent is auditing. Must be passed in the form 0=a0,1=a1 (if agent 0 audits attribute a0 and agent 1 audits attribute a1). If not set, agent x audits the protected attribute at index x.")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "german_credit", "folktables", "propublica"])
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument('--unbias-mean', action='store_true')

    args = parser.parse_args()

    if args.collaborating_agents is not None:
        args.collaborating_agents = [int(aid) for aid in args.collaborating_agents.split(",")]
        assert max(args.collaborating_agents) < args.agents, "Collaborating agent index out of range!"
    else:
        args.collaborating_agents = list(range(args.agents))

    return args

import analyze as az
import os, fnmatch
import argparse


def main(condition, agent_type, seed_num):
    agent_directory = "Agents/{}/{}/{}".format(condition, agent_type, seed_num)
    gen_files = fnmatch.filter(os.listdir(agent_directory), 'gen*')
    gen_numbers = [int(x[3:]) for x in gen_files]
    last_gen = max(gen_numbers)

    if condition == "single":
        if agent_type == "random":
            # random single agents
            td, ag = az.run_random_population(1, "all")
        else:
            # check evolved agents
            az.plot_fitness('single', agent_type, seed_num)
            td, ag = az.run_single_agent('single', agent_type, seed_num, last_gen, 0, "all")
            # az.check_generalization('single', agent_type, seed_num, ag)

            # # additional checks
            # w = az.plot_weights('single', 'buttons', 123, [0, 10, 20, 30, 40], 1)
            # az.animate_trial('single', 'buttons', 123, 0, 0, 0)

    elif condition == "joint":
        if agent_type == "random":
            # random joint agents
            td, a1, a2 = az.run_random_pair(3, 'all')
        else:
            # check evolved joint agents
            # agent_type, seed, generation_num, agent_num, to_plot
            az.plot_fitness('joint', agent_type, seed_num)
            td1, a1, a2 = az.run_single_pair(agent_type, seed_num, last_gen, 0, 'all')


15.0, -14.94964193


if __name__ == '__main__':
    # run with  python simulate.py real > kennylog.txt
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", type=str, help="specify the condition",
                        choices=["single", "joint"])
    parser.add_argument("agent_type", type=str, help="specify the type of the agent you want to run",
                        choices=["buttons", "direct"])
    parser.add_argument("seed_num", type=int, help="specify random seed number")
    args = parser.parse_args()
    main(args.condition, args.agent_type, args.seed_num)


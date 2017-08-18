"""
This is the main file for running evolution of neural network agents in the Knoblich and Jordan (2003) task.
This version does not parallelize the seeds and can be combined with parallel agent processing.
"""
import random
from evolve import Evolution
import json
import argparse
import os
import shutil
# from profilestats import profile
# @profile(print_stats=10, dump_stats=True)


def main(agent_type, seed_num, mutation_variance, prob_crossover):
    # load configuration settings
    json_data = open('config.json')
    config = json.load(json_data)
    json_data.close()

    parent_dir = os.getcwd() + '/Agents/joint/' + agent_type
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    for seed in seed_num:
        # set random seed
        random.seed(seed)

        config['evaluation_params']['velocity_control'] = agent_type

        if mutation_variance:
            config['evolution_params']['mutation_variance'] = mutation_variance
        if prob_crossover:
            config['evolution_params']['prob_crossover'] = prob_crossover

        # set up evolution
        evolution = Evolution(config['evolution_params']['pop_size'],
                              config['evolution_params'],
                              config['network_params'],
                              config['evaluation_params'],
                              config['agent_params'])

        # create the right directory
        foldername = parent_dir + '/' + str(seed)
        evolution.set_foldername(foldername)
        if os.path.exists(foldername):
            shutil.rmtree(foldername)
        os.makedirs(foldername)

        with open(foldername + '/usedconfig.json', 'w') as fp:
            json.dump(config, fp)

        # run evolution from scratch or starting from a given population
        evolution.run_joint(None, parallel_agents=False)
        # evolution.run_joint(None, parallel_agents=True)
        # evolution.run(150)
    return


if __name__ == '__main__':
    # run with  python simulate.py real > kennylog.txt
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_type", type=str, help="specify the type of the agent you want to run",
                        choices=["buttons", "direct"])
    parser.add_argument("seed_num", nargs='+', type=int)
    parser.add_argument("-m", "--mutation_variance", type=int, default=1, help="specify the mutation variance")
    parser.add_argument("-c", "--prob_crossover", type=int, default=0.8, help="specify the probability of crossover")
    args = parser.parse_args()
    main(args.agent_type, args.seed_num, args.mutation_variance, args.prob_crossover)

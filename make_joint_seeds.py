from analyze import load_population, load_config
import os
import pickle
from CTRNN import BrainCTRNN
import agents


config = load_config(None, None, None)


def get_best_agents(agent_type):
    agent_directory = "./Agents/single/" + agent_type
    agent_folders = list(filter(lambda f: not f.startswith('.'), os.listdir(agent_directory)))
    good_agents = []
    for folder in agent_folders:
        seed_files = list(filter(lambda genfile: genfile.startswith('gen'),
                                 os.listdir(agent_directory + '/{}'.format(folder))))
        gen_numbers = [int(x[3:]) for x in seed_files]
        population = load_population('single', agent_type, folder, max(gen_numbers))
        # choose only from seeds in which fitness reached at least 90%
        if population[0].fitness > 0.9:
            good_agents.extend(population[:10])

    # add only unique genotypes
    best_agents = []
    for a in good_agents:
        if a not in best_agents:
            best_agents.append(a)

    return best_agents


# best_buttons = get_best_agents('buttons')
best_direct = get_best_agents('direct')


def create_random_pop(size, agent_type):
    population = []
    for i in range(size):
        # create the agent's CTRNN brain
        agent_brain = BrainCTRNN(config['network_params']['num_neurons'],
                                 config['network_params']['step_size'],
                                 config['network_params']['tau_range'],
                                 config['network_params']['g_range'],
                                 config['network_params']['theta_range'],
                                 config['network_params']['w_range'])

        if agent_type == "direct":
            agent = agents.DirectVelocityAgent(agent_brain, config['agent_params'],
                                               config['evaluation_params']['screen_width'])
        else:
            agent = agents.EmbodiedAgentV2(agent_brain, config['agent_params'],
                                           config['evaluation_params']['screen_width'])
        population.append(agent)
    return population


fillsize = 50 - len(best_direct)
random_fill = create_random_pop(fillsize, 'direct')
best_direct.extend(random_fill)


def save_pop(population, agent_type):
    pop_file = open('./Agents/joint/{}/seedpop'.format(agent_type), 'wb')
    pickle.dump(population, pop_file)
    pop_file.close()


# save_pop(best_buttons, 'buttons')
save_pop(best_direct, 'direct')

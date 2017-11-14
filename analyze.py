import pickle
import simulate
import matplotlib.pyplot as plt
import json
import evolve
import numpy as np
import datetime
import os
# from images2gif import writeGif


def load_config(condition, agent_type, seed):
    if condition is not None:
        json_data = open('./Agents/{}/{}/{}/usedconfig.json'.format(condition, agent_type, seed), 'r')
    else:
        json_data = open('config.json', 'r')
    config = json.load(json_data)
    json_data.close()
    return config


def load_population(condition, agent_type, seed, gen):
    pop_file = open('./Agents/{}/{}/{}/gen{}'.format(condition, agent_type, seed, gen), 'rb')
    population = pickle.load(pop_file)
    pop_file.close()
    if condition == 'single':
        population.sort(key=lambda agent: agent.fitness, reverse=True)
        return population
    else:
        left_pop = population['left']
        right_pop = population['right']
        left_pop.sort(key=lambda agent: agent.fitness, reverse=True)
        right_pop.sort(key=lambda agent: agent.fitness, reverse=True)
        return left_pop, right_pop


def pop_fitness(population):
    return [agent.fitness for agent in population]


def plot_data(trial_data, to_plot, fig_title, lims):

    if to_plot == "all":
        num_trials = len(trial_data['target_pos'])
        num_cols = num_trials/2

        fig = plt.figure(figsize=(10, 6))
        fig.suptitle(fig_title)

        for p in range(num_trials):
            ax = fig.add_subplot(2, num_cols, p+1)
            ax.set_ylim(lims)
            ax.plot(trial_data['target_pos'][p], label='Target position')
            ax.plot(trial_data['tracker_pos'][p], label='Tracker position')
            ax.plot(trial_data['tracker_v'][p], label='Tracker velocity')
            ax.plot(trial_data['keypress'][p][:, 0], label='Left keypress')
            ax.plot(trial_data['keypress'][p][:, 1], label='Right keypress')
            # ax.plot(trial_data['input'][p][:, 0], label='Input to n1')
            # ax.plot(trial_data['input'][p][:, 1], label='Input to n2')
            # ax.plot(trial_data['input'][p][:, 2], label='Input to n3')
            # ax.plot(trial_data['input'][p][:, 3], label='Input to n4')
            # ax.plot(trial_data['input'][p][:, 4], label='Input to n5')
            # ax.plot(trial_data['input'][p][:, 5], label='Input to n6')
            # ax.plot(trial_data['input'][p][:, 6], label='Input to n7')
            # ax.plot(trial_data['input'][p][:, 7], label='Input to n8')
            # ax.plot(trial_data['output'][p][:, 0], label='Output of n7')
            # ax.plot(trial_data['output'][p][:, 1], label='Output of n8')
        plt.legend()
        plt.show()

    elif to_plot == 'none':
        pass

    else:
        plt.plot(trial_data['target_pos'][to_plot], label='Target position')
        plt.plot(trial_data['tracker_pos'][to_plot], label='Tracker position')
        plt.plot(trial_data['tracker_v'][to_plot], label='Tracker velocity')
        plt.plot(trial_data['keypress'][to_plot][:, 0], label='Left keypress')
        plt.plot(trial_data['keypress'][to_plot][:, 1], label='Right keypress')
        plt.plot(trial_data['input'][to_plot][:, 4], label='Input to n5')
        plt.plot(trial_data['input'][to_plot][:, 5], label='Input to n6')

        plt.legend()
        plt.show()


def run_random_population(size, to_plot):
    """
    Creates a population of composed of random agents, run them through the experiment and plot the best one.
    :param size: the size of the population to generate
    :param to_plot: which trials to plot
    :return: 
    """
    config = load_config(None, None, None)
    evolution = evolve.Evolution(config['evolution_params']['pop_size'],
                                 config['evolution_params'],
                                 config['network_params'],
                                 config['evaluation_params'],
                                 config['agent_params'])
    population = evolution.create_population(size)
    for agent in population:
        simulation_run = simulate.Simulation(evolution.step_size, evolution.evaluation_params)
        # simulation_run = simulate.SimpleSimulation(evolution.step_size, evolution.evaluation_params)
        trial_data = simulation_run.run_trials(agent, simulation_run.trials)  # returns a list of fitness in all trials
        # agent.fitness = np.mean(trial_data['fitness'])
        agent.fitness = evolution.harmonic_mean(trial_data['fitness'])
        # agent.fitness = min(trial_data['fitness'])
    population.sort(key=lambda ag: ag.fitness, reverse=True)
    agent = population[0]
    print(agent.fitness)
    simulation_run = simulate.Simulation(evolution.step_size, evolution.evaluation_params)
    # simulation_run = simulate.SimpleSimulation(evolution.step_size, evolution.evaluation_params)
    trial_data = simulation_run.run_trials(agent, simulation_run.trials, savedata=True)  # returns a list of fitness in all trials
    fig_title = "Best agent from random population"
    lims = [config['evaluation_params']['screen_width'][0]-1, config['evaluation_params']['screen_width'][1]+1]
    plot_data(trial_data, to_plot, fig_title, lims)

    return trial_data, agent


def run_single_agent(condition, agent_type, seed, generation_num, agent_num, to_plot):
    """
    Load a specified generation and plot a specified agent behavior (the first one is the best performing).
    :param condition: which condition (single, joint)
    :param agent_type: what kind of agent (buttons, direct)
    :param seed: which seed
    :param generation_num: which generation to use
    :param agent_num: which agent to plot
    :param to_plot: which trials to plot
    :return: 
    """
    config = load_config(condition, agent_type, seed)
    population = load_population(condition, agent_type, seed, generation_num)
    agent = population[agent_num]
    simulation_run = simulate.Simulation(config['network_params']['step_size'], config['evaluation_params'])
    # simulation_run = simulate.SimpleSimulation(config['network_params']['step_size'], config['evaluation_params'])
    trial_data = simulation_run.run_trials(agent, simulation_run.trials, savedata=True)
    fig_title = "Generation {}, agent {}".format(generation_num, agent_num)
    lims = [config['evaluation_params']['screen_width'][0]-1, config['evaluation_params']['screen_width'][1]+1]
    plot_data(trial_data, to_plot, fig_title, lims)

    return trial_data, agent


def plot_fitness(condition, agent_type, seed):
    fit_file = open('./Agents/{}/{}/{}/fitnesses'.format(condition, agent_type, seed), 'rb')
    fits = pickle.load(fit_file)
    fit_file.close()

    plt.plot(fits['average'], label="Average population fitness")
    plt.plot(fits['best'], label="Best agent fitness")

    # if condition == 'single':
    #     plt.plot(fits['average'], label="Average population fitness")
    #     plt.plot(fits['best'], label="Best agent fitness")
    # else:
    #     # plt.plot(fits['average']['combined'], 'ro-', label="Average trial fitness")
    #     plt.plot(fits['average']['left'], 'r.-', label="Average population fitness")
    #     #plt.plot(fits['average']['right'], 'g.-', label="Average right population fitness",)
    #
    #     # plt.plot(fits['best']['combined'], 'ro--', label="Best trial fitness")
    #     plt.plot(fits['best']['left'], 'b.--', label="Best left agent fitness")
    #     plt.plot(fits['best']['right'], 'g.--', label="Best right agent fitness")

    plt.legend()
    plt.title("Fitness over generations")
    plt.show()


def plot_weights(condition, agent_type, seed, gens, agent_num):
    weights = np.zeros((len(gens), 64))
    for i in range(len(gens)):
        trd, ag = run_single_agent(condition, agent_type, seed, gens[i], agent_num, "none")
        weights[i, :] = np.reshape(ag.brain.W, (1, 64))

    plt.plot(weights)
    plt.xticks(list(range(len(gens))), gens)
    plt.show()

    return weights


def animate_trial(condition, agent_type, seed, generation_num, agent_num, trial_num):
    """
    Load a specified generation and create an animation of a specified agent behavior (the first one is the best performing)
    in a specified trial. The animation is created as a sequence of png files that later need to be manually converted
    into gif.
    :param condition: which condition (single, joint)
    :param agent_type: which agent (buttons, direct)
    :param seed: which seed number
    :param generation_num: which generation to use
    :param agent_num: which agent to plot
    :param trial_num: which trials to plot
    :return: 
    """
    config = load_config(condition, agent_type, seed)
    population = load_population(condition, agent_type, seed, generation_num)
    agent = population[agent_num]
    simulation_run = simulate.Simulation(config['network_params']['step_size'], config['evaluation_params'])
    trial_data = simulation_run.run_trials(agent, simulation_run.trials, savedata=True)

    fitness = trial_data['fitness'][trial_num]
    target_pos = trial_data['target_pos'][trial_num]
    tracker_pos = trial_data['tracker_pos'][trial_num]
    # keypress = trial_data['keypress'][trial_num]
    sim_length = simulation_run.sim_length[trial_num] + simulation_run.start_period

    # Save the current state of the system
    times = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Create folder for images
    os.makedirs("./Animations/{}".format(times))

    ticker = 10  # just plot every 10th (x-th) state.
    counter_img = 0
    counter_sec = 0

    # Region Borders
    upper_bound = config['evaluation_params']['screen_width'][1]
    lower_bound = config['evaluation_params']['screen_width'][0]
    screen_size = upper_bound - lower_bound
    region_width = screen_size / 3
    right_border = round(0 + region_width / 2, 2)
    left_border = round(0 - region_width / 2, 2)

    # Set initial target direction:
    if target_pos[simulation_run.start_period] > 0:
        direction = "right"
    else:
        direction = "left"

    y_range = range(-5, 5)

    for i in range(0, sim_length, ticker):

        plt.figure(figsize=(10, 6), dpi=80)

        plt.plot(np.repeat(left_border, len(y_range)), y_range, "--", c="grey", alpha=0.2)  # Region Left
        plt.plot(np.repeat(right_border, len(y_range)), y_range, "--", c="grey", alpha=0.2)  # Region Right

        plt.plot(tracker_pos[i], 0, 'ro', markersize=12, alpha=0.5)  # Tracker
        plt.plot(target_pos[i], 0, 'go')  # Target

        # if any(keypress[i:i+ticker, 0] == -1):
        #     plt.plot(-10, -4, 'bs', markersize=16)  # keypress left
        # if any(keypress[i:i+ticker, 1] == 1):
        #     plt.plot(10, -4, 'bs', markersize=16)  # keypress right

        # if condition == "sound":
        #     if any(sounds[i:i + ticker, 0] == 1):
        #         plt.plot(-10, -3.9, 'yo', markersize=24, alpha=0.3)  # sound left
        #     if any(sounds[i:i + ticker, 1] == 1):
        #         plt.plot(10, -3.9, 'yo', markersize=24, alpha=0.3)  # sound right

        # Define boarders
        plt.xlim(-20, 20)
        plt.ylim(-5, 5)

        # Remove y-Axis
        plt.yticks([])

        # Print fitnesss, time and conditions in plot
        plt.annotate(xy=[0, 4], xytext=[0, 4], s="Trial Fitness: {}".format(round(fitness, 2)))  # Fitness

        # Updated time-counter:
        if counter_img == 25:
            counter_sec += 1
            print("{}% ready".format(np.round((i / sim_length) * 100, 2)))  # gives feedback how much is plotted already.

        counter_img = counter_img + 1 if counter_img < 25 else 1

        # Update simulation time:
        sim_msec = i if i < 100 else i % 100
        sim_sec = int(i * config['network_params']['step_size'])  # or int(i/100)

        plt.annotate(xy=[-15, 4], xytext=[-15, 4], s="{}:{}sec (Real Time)".format(str(counter_sec).zfill(2),
                                                                                   str(counter_img).zfill(2)))  # Real Time

        plt.annotate(xy=[-15, 3.5], xytext=[-15, 3.5], s="{}:{}sec (Simulation Time)".format(str(sim_sec).zfill(2),
                                                                                             str(sim_msec).zfill(2)))  # Simulation Time

        plt.annotate(xy=[0, 3.5], xytext=[0, 3.5], s="Initial Target Direction: {}".format(direction))
        plt.annotate(xy=[0, 3.0], xytext=[0, 3.0], s="Target Speed: {}".format(abs(config['evaluation_params']['velocities'][trial_num])))
        plt.annotate(xy=[-15, 3.0], xytext=[-15, 3.0], s="Sound Condition: {}".format(config['evaluation_params']['condition']))  # condition

        plt.savefig('./Animations/{}/animation{}.png'
                    .format(times,
                            str(int(i / ticker)).zfill(len(str(int(sim_length / ticker))))))

        plt.close()
        print("Animation complete")
# in terminal:
# convert -delay 5 -loop 0 animation*.jpg animated.gif

# images = os.listdir('./Animations/2017-05-16_16-33-18')
# writeGif("images.gif", images, duration=1, dither=0)


def check_generalization(condition, agent_type, seed, agent):
    config = load_config(condition, agent_type, seed)
    config["evaluation_params"]["impacts"] = [0.5, 2]
    simulation_run = simulate.Simulation(config['network_params']['step_size'], config['evaluation_params'])
    trial_data = simulation_run.run_trials(agent, simulation_run.trials, savedata=True)
    fig_title = "Generalization over impact"
    lims = [config['evaluation_params']['screen_width'][0] - 1, config['evaluation_params']['screen_width'][1] + 1]
    plot_data(trial_data, "all", fig_title, lims)

    config["evaluation_params"]["impacts"] = [1]
    config["evaluation_params"]["tg_start_range"] = [-3, 3]
    config["evaluation_params"]["tg_start_variants"] = [2]

    simulation_run = simulate.Simulation(config['network_params']['step_size'], config['evaluation_params'])
    trial_data = simulation_run.run_trials(agent, simulation_run.trials, savedata=True)
    fig_title = "Generalization over starting position"
    lims = [config['evaluation_params']['screen_width'][0] - 1, config['evaluation_params']['screen_width'][1] + 1]
    plot_data(trial_data, "all", fig_title, lims)


def run_random_pair(size, to_plot):
    """
    Creates a population of composed of random agents, run them through the experiment and plot the best one.
    :param size: the size of the population to generate
    :param to_plot: which trials to plot
    :return:
    """
    config = load_config(None, None, None)
    evolution = evolve.Evolution(config['evolution_params']['pop_size'],
                                 config['evolution_params'],
                                 config['network_params'],
                                 config['evaluation_params'],
                                 config['agent_params'])
    population_left = evolution.create_population(size)
    population_right = evolution.create_population(size)

    for agent1 in population_left:
        for agent2 in population_right:
            simulation_run = simulate.Simulation(evolution.step_size, evolution.evaluation_params)
            # simulation_run = simulate.SimpleSimulation(evolution.step_size, evolution.evaluation_params)
            trial_data = simulation_run.run_joint_trials(agent1, agent2, simulation_run.trials)  # returns a list of fitness in all trials
            # agent.fitness = np.mean(trial_data['fitness'])
            trial_fitness = evolution.harmonic_mean(trial_data['fitness'])
            agent1.fitness = agent1.fitness + trial_fitness
            agent2.fitness = agent2.fitness + trial_fitness
            # agent.fitness = min(trial_data['fitness'])
        agent1.fitness = agent1.fitness/size

    for agent in population_right:
        agent.fitness = agent.fitness/size

    population_left.sort(key=lambda ag: ag.fitness, reverse=True)
    population_right.sort(key=lambda ag: ag.fitness, reverse=True)

    agent1 = population_left[0]
    agent2 = population_right[0]

    print(agent1.fitness, agent2.fitness)

    simulation_run = simulate.Simulation(evolution.step_size, evolution.evaluation_params)
    # simulation_run = simulate.SimpleSimulation(evolution.step_size, evolution.evaluation_params)
    # returns a list of fitness in all trials
    trial_data = simulation_run.run_joint_trials(agent1, agent2, simulation_run.trials, savedata=True)
    fig_title = "Best agent pair from random population"
    lims = [config['evaluation_params']['screen_width'][0]-1, config['evaluation_params']['screen_width'][1]+1]
    plot_data(trial_data, to_plot, fig_title, lims)

    return trial_data, agent1, agent2


def run_single_pair(agent_type, seed, generation_num, agent_num, to_plot):
    """
    Load a specified generation and plot a specified agent behavior (the first one is the best performing).
    :param condition: which condition (single, joint)
    :param agent_type: what kind of agent (buttons, direct)
    :param seed: which seed
    :param generation_num: which generation to use
    :param agent_num: which agent to plot
    :param to_plot: which trials to plot
    :return:
    """
    config = load_config('joint', agent_type, seed)
    left_pop, right_pop = load_population('joint', agent_type, seed, generation_num)

    agent1 = left_pop[agent_num]
    agent2 = right_pop[agent_num]

    simulation_run = simulate.Simulation(config['network_params']['step_size'], config['evaluation_params'])
    trial_data = simulation_run.run_joint_trials(agent1, agent2, simulation_run.trials, savedata=True)

    fig_title = "Generation {}, agent {}".format(generation_num, agent_num)
    lims = [config['evaluation_params']['screen_width'][0]-1, config['evaluation_params']['screen_width'][1]+1]
    plot_data(trial_data, to_plot, fig_title, lims)

    return trial_data, agent1, agent2


# def plot_best_pair(agent_type, seed, generation_num, to_plot):
#     """
#     Load a specified generation and plot a specified agent behavior (the first one is the best performing).
#     :param condition: which condition (single, joint)
#     :param agent_type: what kind of agent (buttons, direct)
#     :param seed: which seed
#     :param generation_num: which generation to use
#     :param agent_num: which agent to plot
#     :param to_plot: which trials to plot
#     :return:
#     """
#     config = load_config('joint', agent_type, seed)
#     left_pop, right_pop = load_population('joint', agent_type, seed, generation_num)
#
#     simulation_setup = simulate.Simulation(config['network_params']['step_size'], config['evaluation_params'])
#
#     best_pair_mean = 0
#     a1 = None
#     a2 = None
#
#     for agent1 in left_pop[:10]:
#         for agent2 in right_pop[:10]:
#             trial_data = simulation_setup.run_joint_trials(agent1, agent2, simulation_setup.trials, savedata=True)
#             mean_fitness = np.mean(trial_data['fitness'])
#             if mean_fitness > best_pair_mean:
#                 a1 = agent1
#                 a2 = agent2
#                 best_pair_mean = mean_fitness
#
#     trial_data = simulation_setup.run_joint_trials(a1, a2, simulation_setup.trials, savedata=True)
#     fig_title = "Generation {}, best agent pair".format(generation_num)
#     lims = [config['evaluation_params']['screen_width'][0] - 1,
#             config['evaluation_params']['screen_width'][1] + 1]
#     plot_data(trial_data, to_plot, fig_title, lims)
#
#     return trial_data, a1, a2


def plot_best_pair(agent_type, seed, generation_num, to_plot):
    """
    Load a specified generation and plot a specified agent behavior (the first one is the best performing).
    :param condition: which condition (single, joint)
    :param agent_type: what kind of agent (buttons, direct)
    :param seed: which seed
    :param generation_num: which generation to use
    :param agent_num: which agent to plot
    :param to_plot: which trials to plot
    :return:
    """
    configfile = load_config('joint', agent_type, seed)
    left_pop, right_pop = load_population('joint', agent_type, seed, generation_num)

    simulation_setup = simulate.Simulation(config['network_params']['step_size'], config['evaluation_params'])

    best_pair_mean = 0
    a1 = None
    a2 = None

    for agent1 in left_pop[:10]:
        for agent2 in right_pop[:10]:
            trial_data = simulation_setup.run_joint_trials(agent1, agent2, simulation_setup.trials, savedata=True)
            mean_fitness = np.mean(trial_data['fitness'])
            if mean_fitness > best_pair_mean:
                a1 = agent1
                a2 = agent2
                best_pair_mean = mean_fitness

    trial_data = simulation_setup.run_joint_trials(a1, a2, simulation_setup.trials, savedata=True)
    fig_title = "Generation {}, best agent pair".format(generation_num)
    lims = [config['evaluation_params']['screen_width'][0] - 1,
            config['evaluation_params']['screen_width'][1] + 1]
    plot_data(trial_data, to_plot, fig_title, lims)

    return trial_data, a1, a2

# config = load_config()
# evolution = evolve.Evolution(config['evolution_params']['pop_size'],
#                              config['evolution_params'],
#                              config['network_params'],
#                              config['evaluation_params'],
#                              config['agent_params'])
# population = evolution.create_population(1)
# a = population[0]
# genes = np.zeros([1000, 90])
#
# for i in range(1000):
#     new_a = evolution.mutate(a, 0.1)
#     genes[i,:] = new_a.genotype
#     a = deepcopy(new_a)
# genesavg = np.mean(genes,1)
# plt.plot(genesavg)

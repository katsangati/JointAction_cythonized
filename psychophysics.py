"""
Which analysis will be performed?
1. Plot agent behavior.
1. Neural activation of all 8 neurons throughout the trial.
2. Activation for different types of neurons might need to be plotted separately
    due to the differences in activation range.
3. Activation of neurons compared to neural input they are receiving.
4. Activation of sensory and motor neurons compared to the target-tracker distance.
7. Plot evolved network weights (Hinton diagram).
5. Lesion of auditory information (from the start of the trial vs at specific points).
6. Lesion of visual information (from the start of the trial vs at specific points).
8. Dynamical plots for activation vectors given a particular input.
9. Correlation (mutual information? coupling?) between neural activations of two agents.
10. Prediction between neural activation of one agent and the output of the other.
11. Compare behavioral patterns with the psychological experiment results.

- adding noise
- network lesioning
- pca on neuron activations - see if top components look similar across agents
- cross-correlations
- EEG analysis toolbox in python?
- cross-recurrence
- mutual information
- correlation over all trials or at decision points?
- check generalization again, show that it works
- weight matrix correlation with activation correlation?

sync measures:
imaginary coherence
projected power correlations
for each frequency
"""

import analyze as az
import os
import pickle
import fnmatch
import simulate
import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
from mpl_toolkits.mplot3d import Axes3D


agent_directory = "Agents/single/direct/102575"
gen_files = fnmatch.filter(os.listdir(agent_directory), 'gen*')
gen_numbers = [int(x[3:]) for x in gen_files]
last_gen = max(gen_numbers)

# Plot fitness
# az.plot_fitness("single", "direct", "102575")

# Get trial and agent data
config = az.load_config("single", "direct", "102575")
population = az.load_population("single", "direct", "102575", last_gen)
ag = population[0]

simulation_run = simulate.Simulation(config['network_params']['step_size'], config['evaluation_params'])
td = simulation_run.run_trials(ag, simulation_run.trials, savedata=True)


def plot_data(trial_data, trial_num, to_plot, fig_title):
    if trial_num == "all":
        num_trials = len(trial_data['target_pos'])
        num_cols = num_trials/2

        fig = plt.figure(figsize=(10, 6))
        fig.suptitle(fig_title)

        for p in range(num_trials):
            ax = fig.add_subplot(2, num_cols, p + 1)
            if to_plot == "behavior":
                ax.plot(trial_data['target_pos'][p], label='Target position')
                ax.plot(trial_data['tracker_pos'][p], label='Tracker position')
                ax.plot(trial_data['tracker_v'][p], label='Tracker velocity')
                ax.plot(trial_data['keypress'][p][:, 0], label='Left motor')
                ax.plot(trial_data['keypress'][p][:, 1], label='Right motor')
            elif to_plot == "activation_all":
                ax.plot(trial_data['brain_state'][p][:, 0], label='n1')
                ax.plot(trial_data['brain_state'][p][:, 1], label='n2')
                ax.plot(trial_data['brain_state'][p][:, 2], label='n3')
                ax.plot(trial_data['brain_state'][p][:, 3], label='n4')
                ax.plot(trial_data['brain_state'][p][:, 4], label='n5')
                ax.plot(trial_data['brain_state'][p][:, 5], label='n6')
                ax.plot(trial_data['brain_state'][p][:, 6], label='n7')
                ax.plot(trial_data['brain_state'][p][:, 7], label='n8')
            elif to_plot == "input_all":
                ax.plot(trial_data['input'][p][:, 0], label='n1')
                ax.plot(trial_data['input'][p][:, 1], label='n2')
                ax.plot(trial_data['input'][p][:, 2], label='n3')
                ax.plot(trial_data['input'][p][:, 3], label='n4')
                ax.plot(trial_data['input'][p][:, 4], label='n5')
                ax.plot(trial_data['input'][p][:, 5], label='n6')
                ax.plot(trial_data['input'][p][:, 6], label='n7')
                ax.plot(trial_data['input'][p][:, 7], label='n8')
            elif to_plot == "output_all":
                ax.plot(trial_data['output'][p][:, 0], label='n7')
                ax.plot(trial_data['output'][p][:, 1], label='n8')
    else:
        if to_plot == "behavior":
            plt.plot(trial_data['target_pos'][trial_num], label='Target position')
            plt.plot(trial_data['tracker_pos'][trial_num], label='Tracker position')
            plt.plot(trial_data['tracker_v'][trial_num], label='Tracker velocity')
            plt.plot(trial_data['keypress'][trial_num][:, 0], label='Left motor')
            plt.plot(trial_data['keypress'][trial_num][:, 1], label='Right motor')
        elif to_plot == "activation_all":
            plt.plot(trial_data['brain_state'][trial_num][:, 0], label='n1')
            plt.plot(trial_data['brain_state'][trial_num][:, 1], label='n2')
            plt.plot(trial_data['brain_state'][trial_num][:, 2], label='n3')
            plt.plot(trial_data['brain_state'][trial_num][:, 3], label='n4')
            plt.plot(trial_data['brain_state'][trial_num][:, 4], label='n5')
            plt.plot(trial_data['brain_state'][trial_num][:, 5], label='n6')
            plt.plot(trial_data['brain_state'][trial_num][:, 6], label='n7')
            plt.plot(trial_data['brain_state'][trial_num][:, 7], label='n8')
        elif to_plot == "input_all":
            plt.plot(trial_data['input'][trial_num][:, 0], label='n1')
            plt.plot(trial_data['input'][trial_num][:, 1], label='n2')
            plt.plot(trial_data['input'][trial_num][:, 2], label='n3')
            plt.plot(trial_data['input'][trial_num][:, 3], label='n4')
            plt.plot(trial_data['input'][trial_num][:, 4], label='n5')
            plt.plot(trial_data['input'][trial_num][:, 5], label='n6')
            plt.plot(trial_data['input'][trial_num][:, 6], label='n7')
            plt.plot(trial_data['input'][trial_num][:, 7], label='n8')
        elif to_plot == "output_all":
            plt.plot(trial_data['output'][trial_num][:, 0], label='n7')
            plt.plot(trial_data['output'][trial_num][:, 1], label='n8')

    plt.legend()
    plt.show()


plot_data(td, "all", "behavior", "Trial behavior")
plot_data(td, 0, "activation_all", "Neuronal activation")
plot_data(td, 0, "input_all", "Input to neurons")
plot_data(td, 0, "output_all", "Neuronal output")


def plot_inputs(agent, trial_data, neuron, title):
    num_trials = len(trial_data['target_pos'])
    num_cols = num_trials / 2
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title)

    for p in range(num_trials):
        ax = fig.add_subplot(2, num_cols, p + 1)
        if neuron in range(1,5):
            x = trial_data['input'][p][:, neuron-1]/agent.VW[neuron-1]
        else:
            x = trial_data['input'][p][:, neuron - 1] / agent.AW[neuron - 5]
        y = trial_data['brain_state'][p][:, neuron-1]
        ax.set_xlabel("Input")
        ax.set_ylabel("Activation")
        ax.plot(x, y)
        ax.plot(x[0], y[0], 'ro', markersize=10)
        ax.plot(x[-1], y[-1], 'ro', markersize=10, mfc='none')


plot_inputs(ag, td, 1, "Distance to left border: input vs activation")
plot_inputs(ag, td, 2, "Distance to right border: input vs activation")
plot_inputs(ag, td, 3, "Left eye distance to target: input vs activation")
plot_inputs(ag, td, 4, "Right eye distance to target: input vs activation")
plot_inputs(ag, td, 5, "Left ear: input vs activation")
plot_inputs(ag, td, 6, "Right ear: input vs activation")


def plot_outputs(trial_data, neuron, title):
    num_trials = len(trial_data['target_pos'])
    num_cols = num_trials / 2
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title)

    for p in range(num_trials):
        ax = fig.add_subplot(2, num_cols, p + 1, projection='3d')
        x = trial_data['output'][p][:, 0]
        y = trial_data['output'][p][:, 1]
        z = trial_data['keypress'][p][:, neuron-7]
        ax.set_xlabel("Output of n7")
        ax.set_ylabel("Output of n8")
        ax.set_zlabel("Motor activation")
        ax.plot(x, y, z)


plot_outputs(td, 7, "Motor neuron outputs vs left motor activation")
plot_outputs(td, 8, "Motor neuron outputs vs right motor activation")


def plot_input_output(agent, trial_data, trial_num, title):
    if trial_num == "all":
        num_trials = len(trial_data['target_pos'])
        num_cols = num_trials / 2
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(title)

        for p in range(num_trials):
            ax = fig.add_subplot(2, num_cols, p + 1)
            d = td['target_pos'][p] - td['tracker_pos'][p]
            i1 = trial_data['input'][p][:, 2]/agent.VW[2]
            i2 = trial_data['input'][p][:, 3]/agent.VW[3]
            o1 = trial_data['keypress'][p][:, 0]
            o2 = trial_data['keypress'][p][:, 1]
            ax.plot(d, label="distance to target")
            ax.plot(i1, label="distance to left eye")
            ax.plot(i2, label="distance to right eye")
            ax.plot(o1, label="left motor activation")
            ax.plot(o2, label="right motor activation")
        ax.legend(bbox_to_anchor=(0., 0.), ncol=2, loc=2, borderaxespad=0.)
    else:
        d = td['target_pos'][trial_num] - td['tracker_pos'][trial_num]
        o1 = trial_data['keypress'][trial_num][:, 0]
        o2 = trial_data['keypress'][trial_num][:, 1]
        plt.plot(d, label="distance to target")
        plt.plot(o1, label="left motor activation")
        plt.plot(o2, label="right motor activation")
        plt.legend(bbox_to_anchor=(0., 0.), ncol=2, loc=2, borderaxespad=0.)


plot_input_output(ag, td, 0, "Distance to target vs motor activation")
# peaks

# Object centered motion over time plot: distance to target over time
plt.plot(td['target_pos'][0] - td['tracker_pos'][0])

# the agent's motion over time in response to the target held constantly at different distances in different locations
immobile_test = simulate.SimpleSimulation(config['network_params']['step_size'], config['evaluation_params'])
td_immobile = immobile_test.run_trials(ag, immobile_test.trials, savedata=True)

plot_data(td_immobile, "all", "behavior", "Trial behavior")
plot_data(td_immobile, "all", "activation_all", "Neuronal activation")
plot_data(td_immobile, "all", "input_all", "Input to neurons")
plot_data(td_immobile, "all", "output_all", "Neuronal output")


def linmap(vin, rin, rout):
    """
    Map a vector between 2 ranges.
    :param vin: input vector to be mapped
    :param rin: range of vin to map from
    :param rout: range to map to
    :return: mapped output vector
    :rtype np.ndarray
    """
    a = rin[0]
    b = rin[1]
    c = rout[0]
    d = rout[1]
    return ((c + d) + (d - c) * ((2 * vin - (a + b)) / (b - a))) / 2


def get_params_from_genotype(agent):
    """Take a genotype vector and read off agent parameters."""
    genorest, vw, aw, mw = np.hsplit(agent.genotype, agent.crossover_points[-3:])
    unflattened = genorest.reshape(agent.n_evp + agent.brain.N, agent.brain.N, order='F')
    tau, theta, w = (np.squeeze(a) for a in np.vsplit(unflattened, [1, 2]))
    Tau = linmap(tau, agent.gene_range, config["network_params"]["tau_range"])
    Theta = linmap(theta, agent.gene_range, config["network_params"]["theta_range"])
    W = linmap(w, agent.gene_range, config["network_params"]["w_range"]).transpose()
    return Tau, Theta, W


tau, theta, w = get_params_from_genotype(ag)


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix.
    Positive and negative values are represented by white and black squares, respectively,
    and the size of each square represents the magnitude of each value.
    """
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    if matrix.ndim == 2:
        for (x, y), w in np.ndenumerate(matrix):
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=color)
            ax.add_patch(rect)
    else:
        for (x), w in np.ndenumerate(matrix):
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle([x - size / 2, 1], size, size,
                                 facecolor=color, edgecolor=color)
            ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


hinton(w, max_weight=15)
hinton(tau, max_weight=100)
hinton(theta, max_weight=15)
hinton(ag.VW, max_weight=100)
hinton(ag.AW, max_weight=100)
hinton(ag.MW, max_weight=10)


"""Dynamic plots"""

"""
Define a helper function, which given a point in the state space, will tell us what the derivatives of 
the state elements will be. One way to do this is to run the model over a single timestep, and extract 
the derivative information.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def get_derivative(network, state):
    # compute the next state of the network given its current state and the euler equation
    # update the outputs of all neurons
    o = sigmoid(np.multiply(network.G, state + network.Theta))
    # update the state of all neurons
    dy_dt = np.multiply(1 / network.Tau, - state + np.dot(network.W, o) + network.I) * network.h
    state += dy_dt
    return tuple(dy_dt)
# vget_derivatives = np.vectorize(get_derivative)
"""


def plot_motor_portrait(trial_data, trial_num):
    # Load the activation history of two motor neurons
    net_history = trial_data['brain_state'][trial_num][:, 6:8]
    # Define the sample space (plotting ranges)
    ymin = np.amin(net_history)
    ymax = np.amax(net_history)
    # Define plotting grid
    y1 = np.linspace(ymin, ymax, 30)
    y2 = np.linspace(ymin, ymax, 30)
    Y1, Y2 = np.meshgrid(y1, y2)
    # Load the derivatives
    changes_y1 = trial_data['derivatives'][trial_num][:, 6]
    changes_y2 = trial_data['derivatives'][trial_num][:, 7]

    # Plot the phase portrait
    # quiver function takes the grid of x-y coordinates and their derivatives
    plt.figure(figsize=(10, 6))
    plt.quiver(Y1, Y2, changes_y1, changes_y2, color='b', alpha=.75)
    plt.plot(net_history[:, 0], net_history[:, 1], color='r')
    plt.box('off')
    plt.xlabel('y1', fontsize=14)
    plt.ylabel('y2', fontsize=14)
    plt.title('Phase portrait of two motor neurons', fontsize=16)
    plt.show()


"""
We will plot the phase portraits of 2 motor neurons in different situations:
- constant input from the target position when the tracker is in the middle at the start
- constant input from the target position when the tracker is midway towards the border (approaching)
- constant input from the target position when the tracker is immediately in front of the border
- constant input from the target position when the tracker is right at the border
- constant input from the target position when the tracker is moving away from the border
"""

# constant target input with tracker in the center
plot_motor_portrait(td_immobile, 0)
plot_motor_portrait(td_immobile, 3)

"""
Generalization:
- other target and tracker speeds
- other initial target location
- other boundary size
- plot performance score for these tests
- target reversing before the border
"""

az.check_generalization('single', 'direct', 102575, ag)

"""
Lesion studies:
- no auditory information
- no visual information of the border
- no visual information of the tracker
- loss of perceptual information at critical points
"""

lesion_test = simulate.LesionedSimulation(config['network_params']['step_size'], config['evaluation_params'])
td_deaf_start = lesion_test.run_trials(ag, lesion_test.trials, 100, "auditory", savedata=True)
td_deaf_half = lesion_test.run_trials(ag, lesion_test.trials, 2, "auditory", savedata=True)
plot_data(td_deaf_start, "all", "behavior", "Trial behavior")
plot_data(td_deaf_half, "all", "behavior", "Trial behavior")

td_borderblind_start = lesion_test.run_trials(ag, lesion_test.trials, 100, "visual_border", savedata=True)
td_borderblind_half = lesion_test.run_trials(ag, lesion_test.trials, 2, "visual_border", savedata=True)
plot_data(td_borderblind_start, "all", "behavior", "Trial behavior")
plot_data(td_borderblind_half, "all", "behavior", "Trial behavior")

td_targetblind_start = lesion_test.run_trials(ag, lesion_test.trials, 100, "visual_target", savedata=True)
td_targetblind_half = lesion_test.run_trials(ag, lesion_test.trials, 2, "visual_target", savedata=True)
plot_data(td_targetblind_start, "all", "behavior", "Trial behavior")
plot_data(td_targetblind_half, "all", "behavior", "Trial behavior")


"""
The key question is: does the agent predict the target's reversal at the border or does it merely follow it step
after step? How can we probe this? If the target disappears for some time, will the agent continue moving? Does it
differ depending on whether the target is moving towards the border in a straight stretch vs away from the border
in a straight stretch vs right at the turn before the border?
i.e. if the agent is following the target in a straight direction and will continue moving as the target disappears,
we can say the agent "predicts" the target's location (this is more like emulation);
what happens if the target reappears at a shorter distance than predicted? or at a longer one?
is the agent "surprised"? does it try to compensate or does it simply keep moving?
what if we test this right before the border turn? 
"""


"""JOINT"""

agent_directory = "Agents/joint/direct/205755"
gen_files = fnmatch.filter(os.listdir(agent_directory), 'gen*')
gen_numbers = [int(x[3:]) for x in gen_files]
last_gen = max(gen_numbers)

# Plot fitness
# az.plot_fitness("single", "direct", "102575")

# Get trial and agent data
config = az.load_config("joint", "direct", "205755")
left, right = az.load_population("joint", "direct", "205755", last_gen)
ag1 = left[0]
ag2 = right[0]

simulation_run = simulate.Simulation(config['network_params']['step_size'], config['evaluation_params'])
tdj = simulation_run.run_joint_trials(ag1, ag2, simulation_run.trials, savedata=True)


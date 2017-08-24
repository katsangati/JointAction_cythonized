import numpy as np
cimport numpy as np


cdef class BrainCTRNN:

    def __init__(self, number_of_neurons=0, step_size=0.01, tau_range=(1, 1), gain_range=(1, 1), theta_range=(0, 0), w_range=(0, 0)):
        """
        Initialize a fully connected CTRNN of size N (number_of_neurons) with the following attributes:
        Y      = 'state of each neuron' at current time point i
        Tau    = 'time constant (tau > 0)'
        W     = 'fixed strength of the connection from jth to ith neuron', Weight Matrix
        Theta = 'the bias term'
        sigma = 'the sigmoid function / standard logistic activation function' 1/(1+np.exp(-x))
        I     = 'constant external input' at current time point i
        G     = 'gain' (makes neurons highly sensitive to their input, primarily for motor or sensory nodes)
                 Preferably g is between [1,5] and just > 1 for neurons connected to sensory input or motor output.
        :param number_of_neurons: number of neurons in the network
        :param step_size: step size for the update function
        :param tau: time constants
        :param gain: gains
        :param theta: bias terms
        :param weights: weights
        :return: output
        """

        self.step_size = step_size
        self.N = number_of_neurons
        self.Y = np.zeros(self.N)
        self.dy_dt = np.zeros(self.N)
        self.I = np.zeros(self.N)
        self.tau_range = tau_range
        self.g_range = gain_range
        self.theta_range = theta_range
        self.w_range = w_range
        # In the random searches, the initial population is generated by randomly assigning parameter values drawn from
        # a uniform distribution over the allowed ranges
        self.Tau = np.random.uniform(self.tau_range[0], self.tau_range[1], self.N)
        self.G = np.random.uniform(self.g_range[0], self.g_range[1], self.N)
        self.W = np.random.uniform(self.w_range[0], self.w_range[1], (self.N, self.N))
        # self.Theta = np.random.uniform(self.theta_range[0], self.theta_range[1], self.N)
        self.Theta = center_cross(self.W)
        # self.genotype = self.make_genotype_from_params()  # these are the evolvable parameters

    def randomize_state(self, state_range):
        # To start the simulation it is often useful to randomize initial neuron activation around 0
        self.Y = np.random.uniform(state_range[0], state_range[1], self.N)

    def euler_step(self):
        # Compute the next state of the network given its current state and the simple euler equation
        # update the outputs of all neurons
        o = sigmoid(np.multiply(self.G, self.Y + self.Theta))
        # update the state of all neurons
        self.dy_dt = np.multiply(1 / self.Tau, - self.Y + np.dot(self.W, o) + self.I) * self.step_size
        self.Y += self.dy_dt

    def get_state(self):
        return self.Y


cdef center_cross(np.ndarray weights):
    theta = -np.sum(weights, axis=1)/2
    return theta


cdef sigmoid(np.ndarray x):
    return 1 / (1 + np.exp(-x))
import numpy as np
import random
import string


class Simulation:
    def __init__(self, step_size, evaluation_params):
        self.width = evaluation_params['screen_width']  # [-20, 20]
        self.step_size = step_size  # how fast things are happening in the simulation
        # self.trials = self.create_trials(evaluation_params['velocities'], evaluation_params['impacts'])
        self.trials = self.create_trials(evaluation_params['velocities'], evaluation_params['impacts'],
                                         evaluation_params['tg_start_range'], evaluation_params['tg_start_variants'])
        distance = (self.width[1]-self.width[0]) * evaluation_params['n_turns']  # total distance travelled by target
        # simulation length depends on target velocity
        self.sim_length = [int(distance/abs(trial[0])/self.step_size) for trial in self.trials]
        # self.sim_length = [500] * len(self.trials)
        self.condition = evaluation_params['condition']  # is it a sound condition?
        # the period of time at the beginning of the trial in which the target stays still
        self.start_period = evaluation_params['start_period']
        self.initial_state = evaluation_params['initial_state']
        self.velocity_control = evaluation_params['velocity_control']

    # @staticmethod
    # def create_trials(velocities, impacts):
    #     """
    #     Create a list of trials the environment will run.
    #     :return:
    #     """
    #     trials = [(x, y) for x in velocities for y in impacts]
    #     return trials

    @staticmethod
    def create_trials(velocities, impacts, start_range, size):
        """
        Create a list of trials the environment will run.
        :return:
        """
        if start_range[1]-start_range[0] == 0:
            trials = [(x, y, start_range[0]) for x in velocities for y in impacts]
        else:
            left_positions = np.random.choice(np.arange(start_range[0], 1), size, replace=False)
            right_positions = np.random.choice(np.arange(0, start_range[1] + 1), size, replace=False)
            target_positions = np.concatenate((left_positions, right_positions))
            trials = [(x, y, z) for x in velocities for y in impacts for z in target_positions]
        return trials

    def run_trials(self, agent, trials, savedata=False):
        """
        An evaluation function that accepts an agent and returns a real number representing
        the performance of that parameter vector on the task. Here the task is the Knoblich and Jordan task.

        :param agent: an agent with a CTRNN brain and particular anatomy
        :param trials: a list of trials to perform
        :param savedata: should all the trial data be saved
        :return: fitness
        """

        trial_data = dict()
        trial_data['fitness'] = []
        trial_data['target_pos'] = [None] * len(trials)
        trial_data['tracker_pos'] = [None] * len(trials)
        trial_data['tracker_v'] = [None] * len(trials)
        trial_data['keypress'] = [None] * len(trials)

        if savedata:
            trial_data['brain_state'] = [None] * len(trials)
            trial_data['input'] = [None] * len(trials)
            trial_data['output'] = [None] * len(trials)
            trial_data['button_state'] = [None] * len(trials)

        for i in range(len(trials)):
            # create target and tracker
            # target = Target(trials[i][0], self.step_size, 0)
            target = Target(trials[i][0], self.step_size, trials[i][2])

            if self.velocity_control == "buttons":
                tracker = Tracker(trials[i][1], self.step_size, self.condition)
            elif self.velocity_control == "direct":
                tracker = DirectTracker(None, self.step_size, self.condition)

            # set initial state in specified range
            agent.brain.randomize_state(self.initial_state)
            agent.initialize_buttons()

            trial_data['target_pos'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['tracker_pos'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['tracker_v'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['keypress'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))

            if savedata:
                trial_data['brain_state'][i] = np.zeros((self.sim_length[i] + self.start_period, agent.brain.N))
                trial_data['input'][i] = np.zeros((self.sim_length[i] + self.start_period, agent.brain.N))
                trial_data['output'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))
                trial_data['button_state'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))

            if self.start_period > 0:
                # don't move the target and don't allow tracker to move
                for j in range(self.start_period):
                    agent.visual_input(tracker.position, target.position)
                    agent.brain.euler_step()
                    # activation, motor_activity = agent.motor_output()
                    # activation = agent.motor_output()
                    # tracker.accelerate(activation)

                    trial_data['target_pos'][i][j] = target.position
                    trial_data['tracker_pos'][i][j] = tracker.position
                    trial_data['tracker_v'][i][j] = tracker.velocity
                    # trial_data['keypress'][i][j] = activation

                    if savedata:
                        trial_data['brain_state'][i][j] = agent.brain.Y
                        trial_data['input'][i][j] = agent.brain.I
                        # trial_data['output'][i][j] = motor_activity
                        trial_data['button_state'][i][j] = agent.button_state

            for j in range(self.start_period, self.sim_length[i] + self.start_period):

                # 1) Target movement
                target.movement(self.width)
                # target.movement([self.width[0] + 10, self.width[1] - 10])

                # 2) Agent sees
                agent.visual_input(tracker.position, target.position)

                # 3) Agents moves
                sound_output = tracker.movement(self.width)

                # 4) Agent hears
                if self.condition == 'sound':
                    agent.auditory_input(sound_output)

                trial_data['target_pos'][i][j] = target.position
                trial_data['tracker_pos'][i][j] = tracker.position
                trial_data['tracker_v'][i][j] = tracker.velocity

                if savedata:
                    trial_data['brain_state'][i][j] = agent.brain.Y

                # 5) Update agent's neural system
                agent.brain.euler_step()

                # 6) Agent reacts
                # activation, motor_activity = agent.motor_output()
                activation = agent.motor_output()
                tracker.accelerate(activation)
                # this will save -1 or 1 for button-controlling agents
                # but left and right velocities for direct velocity control agent
                trial_data['keypress'][i][j] = activation

                if savedata:

                    trial_data['input'][i][j] = agent.brain.I
                    # trial_data['output'][i][j] = motor_activity
                    trial_data['button_state'][i][j] = agent.button_state

            # 6) Fitness tacking:
            fitness = 1 - (np.sum(np.abs(trial_data['target_pos'][i] - trial_data['tracker_pos'][i])) /
                           (2 * self.width[1] * (self.sim_length[i] + self.start_period)))
            # penalty for not moving in the trial not counting the delay period
            penalty = list(trial_data['tracker_v'][i][self.start_period:]).count(0) / (self.sim_length[i])
            # if penalty decreases the score below 0, set it to 0
            overall_fitness = np.clip(fitness - penalty, 0, 1)
            trial_data['fitness'].append(overall_fitness)

            # trial_data['fitness'].append(np.mean(trial_data['keypress'][i]))

            # cap_distance = 10
            # total_dist = np.abs(trial_data['target_pos'][i] - trial_data['tracker_pos'][i])
            # scores = np.clip(-1/cap_distance * total_dist + 1, 0, 1)
            # trial_data['fitness'].append(np.mean(scores))
            # scores.sort(reverse=True)
            # trial_data['fitness'].append(np.mean(weighted_scores))

        return trial_data

    def run_joint_trials(self, agent1, agent2, trials, savedata=False):
        """
        An evaluation function that accepts two agents and returns a real number representing
        the performance of these agents on the task.
        Here the task is the Knoblich and Jordan task, the joint action version.

        :param agent1: an agent with a CTRNN brain and particular anatomy, controls only left button
        :param agent2: an agent with a CTRNN brain and particular anatomy, controls only right button
        :param trials: a list of trials to perform
        :param savedata: should all the trial data be saved
        :return: fitness
        """

        trial_data = dict()
        trial_data['fitness'] = []
        trial_data['target_pos'] = [None] * len(trials)
        trial_data['tracker_pos'] = [None] * len(trials)
        trial_data['tracker_v'] = [None] * len(trials)
        trial_data['keypress'] = [None] * len(trials)

        if savedata:
            trial_data['brain_state_a1'] = [None] * len(trials)
            trial_data['input_a1'] = [None] * len(trials)
            trial_data['brain_state_a2'] = [None] * len(trials)
            trial_data['input_a2'] = [None] * len(trials)

            trial_data['output'] = [None] * len(trials)
            trial_data['button_state_a1'] = [None] * len(trials)
            trial_data['button_state_a2'] = [None] * len(trials)

        for i in range(len(trials)):
            # create target and tracker
            # target = Target(trials[i][0], self.step_size, 0)
            target = Target(trials[i][0], self.step_size, trials[i][2])

            if self.velocity_control == "buttons":
                tracker = Tracker(trials[i][1], self.step_size, self.condition)
            elif self.velocity_control == "direct":
                tracker = DirectTracker(None, self.step_size, self.condition)

            # set initial state in specified range
            agent1.brain.randomize_state(self.initial_state)
            agent1.initialize_buttons()

            agent2.brain.randomize_state(self.initial_state)
            agent2.initialize_buttons()

            trial_data['target_pos'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['tracker_pos'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['tracker_v'][i] = np.zeros((self.sim_length[i] + self.start_period, 1))
            trial_data['keypress'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))

            if savedata:
                trial_data['brain_state_a1'][i] = np.zeros((self.sim_length[i] + self.start_period, agent1.brain.N))
                trial_data['input_a1'][i] = np.zeros((self.sim_length[i] + self.start_period, agent1.brain.N))
                trial_data['brain_state_a2'][i] = np.zeros((self.sim_length[i] + self.start_period, agent2.brain.N))
                trial_data['input_a2'][i] = np.zeros((self.sim_length[i] + self.start_period, agent2.brain.N))

                trial_data['output'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))
                trial_data['button_state_a1'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))
                trial_data['button_state_a2'][i] = np.zeros((self.sim_length[i] + self.start_period, 2))

            if self.start_period > 0:
                # don't move the target and don't allow tracker to move
                for j in range(self.start_period):
                    agent1.visual_input(tracker.position, target.position)
                    agent1.brain.euler_step()
                    agent2.visual_input(tracker.position, target.position)
                    agent2.brain.euler_step()

                    # activation, motor_activity = agent.motor_output()
                    # activation = agent.motor_output()
                    # tracker.accelerate(activation)

                    trial_data['target_pos'][i][j] = target.position
                    trial_data['tracker_pos'][i][j] = tracker.position
                    trial_data['tracker_v'][i][j] = tracker.velocity
                    # trial_data['keypress'][i][j] = activation

                    if savedata:
                        trial_data['brain_state_a1'][i][j] = agent1.brain.Y
                        trial_data['input_a1'][i][j] = agent1.brain.I
                        trial_data['brain_state_a2'][i][j] = agent2.brain.Y
                        trial_data['input_a2'][i][j] = agent2.brain.I

                        # trial_data['output'][i][j] = motor_activity
                        trial_data['button_state_a1'][i][j] = agent1.button_state
                        trial_data['button_state_a2'][i][j] = agent2.button_state

            for j in range(self.start_period, self.sim_length[i] + self.start_period):

                # 1) Target movement
                target.movement(self.width)
                # target.movement([self.width[0] + 10, self.width[1] - 10])

                # 2) Agent sees
                agent1.visual_input(tracker.position, target.position)
                agent2.visual_input(tracker.position, target.position)

                # 3) Agents moves
                sound_output = tracker.movement(self.width)

                # 4) Agent hears
                if self.condition == 'sound':
                    agent1.auditory_input(sound_output)
                    agent2.auditory_input(sound_output)

                trial_data['target_pos'][i][j] = target.position
                trial_data['tracker_pos'][i][j] = tracker.position
                trial_data['tracker_v'][i][j] = tracker.velocity

                if savedata:
                    trial_data['brain_state_a1'][i][j] = agent1.brain.Y
                    trial_data['brain_state_a2'][i][j] = agent2.brain.Y

                # 5) Update agent's neural system
                agent1.brain.euler_step()
                agent2.brain.euler_step()

                # 6) Agent reacts
                # activation, motor_activity = agent.motor_output()
                activation_left = agent1.motor_output()[0]  # take only left activation from "left" agent
                activation_right = agent2.motor_output()[1]  # take only right activation from "right" agent

                activation = [activation_left, activation_right]
                tracker.accelerate(activation)
                # this will save -1 or 1 for button-controlling agents
                # but left and right velocities for direct velocity control agent
                trial_data['keypress'][i][j] = activation

                if savedata:
                    trial_data['input_a1'][i][j] = agent1.brain.I
                    # trial_data['output'][i][j] = motor_activity
                    trial_data['button_state_a1'][i][j] = agent1.button_state
                    trial_data['input_a2'][i][j] = agent2.brain.I
                    # trial_data['output'][i][j] = motor_activity
                    trial_data['button_state_a2'][i][j] = agent2.button_state

            # 6) Fitness tacking:
            fitness = 1 - (np.sum(np.abs(trial_data['target_pos'][i] - trial_data['tracker_pos'][i])) /
                           (2 * self.width[1] * (self.sim_length[i] + self.start_period)))
            # penalty for not moving in the trial not counting the delay period
            penalty = list(trial_data['tracker_v'][i][self.start_period:]).count(0) / (self.sim_length[i])
            # if penalty decreases the score below 0, set it to 0
            overall_fitness = np.clip(fitness - penalty, 0, 1)
            trial_data['fitness'].append(overall_fitness)

            # trial_data['fitness'].append(np.mean(trial_data['keypress'][i]))

            # cap_distance = 10
            # total_dist = np.abs(trial_data['target_pos'][i] - trial_data['tracker_pos'][i])
            # scores = np.clip(-1/cap_distance * total_dist + 1, 0, 1)
            # trial_data['fitness'].append(np.mean(scores))
            # scores.sort(reverse=True)
            # trial_data['fitness'].append(np.mean(weighted_scores))

        return trial_data


class SimpleSimulation:
    """
    This is a class that implements a simple simulation version, in which the target appears at a random position
    and remains immobile for the duration of the trial. The tracker's task is to approach the target and stay close.
    """
    def __init__(self, step_size, evaluation_params):
        self.width = evaluation_params['screen_width']  # [-20, 20]
        self.step_size = step_size  # how fast things are happening in the simulation
        self.trials = self.create_trials(5)
        self.sim_length = 1000
        self.condition = evaluation_params['condition']  # is it a sound condition?
        # the period of time at the beginning of the trial in which the target stays still
        self.initial_state = evaluation_params['initial_state']
        self.velocity_control = evaluation_params['velocity_control']

    @staticmethod
    def create_trials(size):
        """
        Create a list of trials the environment will run.
        :return: 
        """
        left_positions = np.random.choice(np.arange(-20, 0), size, replace=False)
        right_positions = np.random.choice(np.arange(1, 21), size, replace=False)
        target_positions = np.concatenate((left_positions, right_positions))
        return target_positions

    def run_trials(self, agent, trials, savedata=False):
        """
        An evaluation function that accepts an agent and returns a real number representing
        the performance of that parameter vector on the task. Here the task is the Knoblich and Jordan task.

        :param agent: an agent with a CTRNN brain and particular anatomy
        :param trials: a list of trials to perform
        :param savedata: should the trial data be saved
        :return: fitness
        """

        trial_data = dict()
        trial_data['fitness'] = []
        trial_data['target_pos'] = [None] * len(trials)
        trial_data['tracker_pos'] = [None] * len(trials)
        trial_data['tracker_v'] = [None] * len(trials)
        trial_data['keypress'] = [None] * len(trials)

        if savedata:
            trial_data['brain_state'] = [None] * len(trials)
            trial_data['input'] = [None] * len(trials)
            trial_data['output'] = [None] * len(trials)
            trial_data['button_state'] = [None] * len(trials)

        for i in range(len(trials)):
            # create target and tracker
            target = ImmobileTarget(trials[i])
            if self.velocity_control == "buttons":
                tracker = Tracker(1, self.step_size, self.condition)
            elif self.velocity_control == "direct":
                tracker = DirectTracker(None, self.step_size, self.condition)
            # set initial state in specified range
            agent.brain.randomize_state(self.initial_state)
            agent.initialize_buttons()

            trial_data['target_pos'][i] = np.zeros((self.sim_length, 1))
            trial_data['tracker_pos'][i] = np.zeros((self.sim_length, 1))
            trial_data['tracker_v'][i] = np.zeros((self.sim_length, 1))
            trial_data['keypress'][i] = np.zeros((self.sim_length, 2))

            if savedata:
                trial_data['brain_state'][i] = np.zeros((self.sim_length, agent.brain.N))
                trial_data['input'][i] = np.zeros((self.sim_length, agent.brain.N))
                trial_data['output'][i] = np.zeros((self.sim_length, 2))
                trial_data['button_state'][i] = np.zeros((self.sim_length, 2))

            for j in range(self.sim_length):

                # 2) Agent sees
                agent.visual_input(tracker.position, target.position)

                # 3) Agents moves
                sound_output = tracker.movement(self.width)

                # 4) Agent hears
                if self.condition == 'sound':
                    agent.auditory_input(sound_output)

                trial_data['target_pos'][i][j] = target.position
                trial_data['tracker_pos'][i][j] = tracker.position
                trial_data['tracker_v'][i][j] = tracker.velocity

                if savedata:
                    trial_data['brain_state'][i][j] = agent.brain.Y

                # 5) Update agent's neural system
                agent.brain.euler_step()

                # 6) Agent reacts
                # activation, motor_activity = agent.motor_output()
                activation = agent.motor_output()
                tracker.accelerate(activation)
                # this will save -1 or 1 for button-controlling agents
                # but left and right velocities for direct velocity control agent
                trial_data['keypress'][i][j] = activation

                if savedata:

                    trial_data['input'][i][j] = agent.brain.I
                    # trial_data['output'][i][j] = motor_activity
                    trial_data['button_state'][i][j] = agent.button_state

            # 6) Fitness tacking:
            fitness = 1 - (np.sum(np.abs(trial_data['target_pos'][i] - trial_data['tracker_pos'][i])) /
                           (2*self.width[1]*self.sim_length))
            # penalty for not moving in the trial not counting the delay period
            penalty = list(trial_data['tracker_v'][i]).count(0)/self.sim_length
            # if penalty decreases the score below 0, set it to 0
            overall_fitness = np.clip(fitness - penalty, 0, 1)
            trial_data['fitness'].append(overall_fitness)

        return trial_data


class ImmobileTarget:
    def __init__(self, position):
        self.position = position


class Target:
    """
    Target moves with constant velocity and starts from the middle.
    Target velocity varies across the trials and can be set to negative or positive values, in which case
    the target's initial movement direction is left or right respectively.
    """

    def __init__(self, velocity, step_size, start_pos):
        self.position = start_pos
        self.velocity = velocity
        self.step_size = step_size

    def reverse_direction(self, border_range, future_pos):
        """
        Reverse target direction if going beyond the border.
        :param border_range: border positions of the environment
        :param future_pos: predicted position at the next time step
        :return: 
        """
        if ((self.velocity > 0) and (future_pos > border_range[1])) or \
                ((self.velocity < 0) and (future_pos < border_range[0])):
            self.velocity *= -1

    def movement(self, border_range):
        future_pos = self.position + self.velocity * self.step_size
        self.reverse_direction(border_range, future_pos)
        self.position += self.velocity * self.step_size


class Tracker:
    """
    Tracker moves as a result of its set velocity and can accelerate based on agent button clicks.
    It starts in the middle of the screen and with initial 0 velocity.
    """

    def __init__(self, impact, step_size, condition):
        self.position = 0
        self.velocity = 0
        self.impact = impact  # how much acceleration is added by button click
        self.step_size = step_size
        # Timer for the emitted sound-feedback
        self.condition = condition  # is it a sound condition?
        self.timer_sound_l = 0
        self.timer_sound_r = 0

    def movement(self, border_range):
        """ Update self.position and self.timer(sound) """

        self.position += self.velocity * self.step_size

        # Tacker does not continue moving, when at the edges of the environment.
        if self.position < border_range[0]:
            self.position = border_range[0]
            self.velocity = 0  # added by GK
        if self.position > border_range[1]:
            self.position = border_range[1]
            self.velocity = 0  # added by GK

        sound_output = [0, 0]

        if self.timer_sound_l > 0:
            self.timer_sound_l -= self.step_size
            sound_output[0] = 1   # auditory feedback from the left button click

        if self.timer_sound_r > 0:
            self.timer_sound_r -= self.step_size
            sound_output[1] = 1  # auditory feedback from the right button click

        return sound_output

    def accelerate(self, inputs):
        """
        Accelerates the tracker to the left or to the right
        Impact of the keypress (how much the velocity is changed) is controlled
        by the tracker's impact attribute.
        :param inputs: an array of size two with values of -1 or +1 (left or right)
        :return: update self.velocity
        """
        acceleration = np.dot(np.array([self.impact, self.impact]).T, inputs)
        self.velocity += acceleration

        if self.condition == "sound":
            self.set_timer(inputs)

    def set_timer(self, left_or_right):
        """ Emit tone of 100-ms duration """
        if left_or_right[0] == -1:  # left
            self.timer_sound_l = 0.1
        if left_or_right[1] == 1:   # right
            self.timer_sound_r = 0.1


class DirectTracker(Tracker):
    """
    DirectTracker moves as a result of its set velocity, which can change based on agent's output
    to left and right motors.
    It starts in the middle of the screen and with initial 0 velocity.
    """

    def __init__(self, impact, step_size, condition):
        Tracker.__init__(self, impact, step_size, condition)

    def accelerate(self, inputs):
        """
        Sets the tracker velocity depending on the activation of the agent's motor neurons.
        The overall velocity is a difference between left and right velocities.
        Whenever the right velocity is greater than left, the tracker moves right and vice versa.
        :param inputs: an array of size two with activation values
        :return: update self.velocity
        """
        new_velocity = inputs[1] - inputs[0]
        # velocity_change = new_velocity - self.velocity

        if self.condition == "sound":
            # self.set_timer(velocity_change)
            self.set_timer(inputs)

        self.velocity = new_velocity

    # def set_timer(self, velocity_change):
    #     """ Emit tone of 100-ms duration """
    #     if velocity_change > 0:
    #         self.timer_sound_r = 0.1
    #     elif velocity_change < 0:
    #         self.timer_sound_l = 0.1

    # def set_timer(self, inputs):
    #     """ Emit tone of 100-ms duration """
    #     if inputs[1] > inputs[0]:  # are we trying to move right?
    #         self.timer_sound_r = 0.1
    #     elif inputs[1] < inputs[0]:
    #         self.timer_sound_l = 0.1
    #     else:
    #         pass

    def set_timer(self, inputs):
        """ Emit tones proportional to the motor activation """
        self.timer_sound_l = inputs[0] * 0.1
        self.timer_sound_r = inputs[1] * 0.1

    def movement(self, border_range):
        """ Update self.position and self.timer(sound) """

        self.position += self.velocity * self.step_size

        # Tacker does not continue moving, when at the edges of the environment.
        if self.position < border_range[0]:
            self.position = border_range[0]
            self.velocity = 0  # added by GK
        if self.position > border_range[1]:
            self.position = border_range[1]
            self.velocity = 0  # added by GK

        sound_output = [self.timer_sound_l, self.timer_sound_r]

        return sound_output


class Agent:
    """
    This is a class that implements agents in the simulation. Agents' brains are CTRNN, but they also have
    a particular anatomy and a connection to external input and output.
    The anatomy is set up by the parameters defined in the config.json file: they specify the number of particular
    sensors and effectors the agent has and the number of connections each sensor has to other neurons, as well
    as the weight and gene ranges.
    The agent initially used in the project, whose perception consisted of absolute positions of target and tracker)
    was defined as follows:
    "agent_params": {"n_visual_sensors": 2, "n_audio_sensors": 2, "n_effectors": 2, "n_visual_connections": 2,
    "n_audio_connections": 2, "n_effector_connections": 2, "gene_range": [0, 1], "evolvable_params": ["tau", "theta"],
    "r_range": [-15, 15], "e_range": [-15, 15]}
    See network.pdf for a picture of that agent.
    """
    def __init__(self, network, agent_parameters):

        self.brain = network
        self.r_range = agent_parameters['r_range']  # receptor gain range
        self.e_range = agent_parameters['e_range']  # effector gain range

        self.VW = np.random.uniform(self.r_range[0], self.r_range[1],
                                    (agent_parameters['n_visual_sensors'] * agent_parameters['n_visual_connections']))
        self.AW = np.random.uniform(self.r_range[0], self.r_range[1],
                                    (agent_parameters['n_audio_sensors'] * agent_parameters['n_audio_connections']))
        self.MW = np.random.uniform(self.e_range[0], self.e_range[1],
                                    (agent_parameters['n_effectors'] * agent_parameters['n_effector_connections']))
        self.gene_range = agent_parameters['gene_range']
        self.genotype = self.make_genotype_from_params()
        self.fitness = 0
        self.n_io = len(self.VW) + len(self.AW) + len(self.MW)  # how many input-output weights

        self.timer_motor_l = 0
        self.timer_motor_r = 0

        # calculate crossover points
        self.n_evp = len(agent_parameters['evolvable_params'])  # how many parameters in addition to weights are evolved
        crossover_points = [i * (self.n_evp + self.brain.N) for i in range(1, self.brain.N + 1)]
        crossover_points.extend([crossover_points[-1] + len(self.VW),
                                 crossover_points[-1] + len(self.VW) + len(self.AW)])
        self.crossover_points = crossover_points
        self.button_state = [False, False]  # both buttons off in the beginning
        self.name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

    def __eq__(self, other):
        # two agents are the same if they have the same genotype
        if np.all(self.genotype == other.genotype):
            return True

    def initialize_buttons(self):
        self.button_state = [False, False]

    def make_genotype_from_params(self):
        """
        Combine all parameters and reshape into a single vector
        :return: [Tau_n1, G_n1, Theta_n1, W_n1..., all visual w, all auditory w, all motor w] 
        """
        # return [self.Tau, self.G, self.Theta, self.W]
        tau = self.linmap(self.brain.Tau, self.brain.tau_range, self.gene_range)
        # skip G in evolution
        # g = self.linmap(self.brain.G, self.brain.g_range, [0, 1])
        theta = self.linmap(self.brain.Theta, self.brain.theta_range, self.gene_range)
        w = self.linmap(self.brain.W.T, self.brain.w_range, self.gene_range)
        vw = self.linmap(self.VW, self.r_range, self.gene_range)
        aw = self.linmap(self.AW, self.r_range, self.gene_range)
        mw = self.linmap(self.MW, self.e_range, self.gene_range)

        stacked = np.vstack((tau, theta, w))
        flattened = stacked.reshape(stacked.size, order='F')
        genotype = np.hstack((flattened, vw, aw, mw))
        return genotype

    def make_params_from_genotype(self, genotype):
        """
        Take a genotype vector and set all agent parameters.
        :param genotype: vector that represents agent parameters in the unified gene range.
        :return: 
        """
        genorest, vw, aw, mw = np.hsplit(genotype, self.crossover_points[-3:])
        self.VW = self.linmap(vw, self.gene_range, self.r_range)
        self.AW = self.linmap(aw, self.gene_range, self.r_range)
        self.MW = self.linmap(mw, self.gene_range, self.e_range)

        unflattened = genorest.reshape(self.n_evp+self.brain.N, self.brain.N, order='F')
        tau, theta, w = (np.squeeze(a) for a in np.vsplit(unflattened, [1, 2]))
        self.brain.Tau = self.linmap(tau, self.gene_range, self.brain.tau_range)
        self.brain.Theta = self.linmap(theta, self.gene_range, self.brain.theta_range)
        self.brain.W = self.linmap(w, self.gene_range, self.brain.w_range).transpose()

    def visual_input(self, position_tracker, position_target):
        """
        The visual input to the agent
        :param position_tracker: absolute position of the tracker 
        :param position_target: absolute position of the target
        :return:
        """
        # add noise to the visual input
        position_tracker = self.add_noise(position_tracker)
        position_target = self.add_noise(position_target)

        self.brain.I[1] = self.VW[0] * position_target  # to n2
        self.brain.I[2] = self.VW[1] * position_tracker  # to n3
        self.brain.I[0] = self.VW[2] * position_target + self.VW[3] * position_tracker  # to n1

    def auditory_input(self, sound_input):
        """
        The auditory input to the agent
        :param sound_input: Tone(s) induced by left and/or right click
        """
        left_click, right_click = sound_input[0], sound_input[1]

        self.brain.I[3] = self.AW[0] * left_click  # to n4
        self.brain.I[5] = self.AW[1] * right_click  # to n6
        self.brain.I[4] = self.AW[2] * left_click + self.AW[3] * right_click  # to n5

    def motor_output(self):
        """
        The motor output of the agent
        :return: output
        """
        # Set activation threshold
        activation = [0, 0]  # Initial activation is zero
        # threshold = 0  # Threshold for output
        threshold = 0.5  # Threshold for output

        # consider adding noise to output before multiplying by motor gains,
        # drawn from a Gaussian distribution with (mu=0, var=0.05)
        o7 = self.brain.Y[6] + self.brain.Theta[6]  # output of n7
        o8 = self.brain.Y[7] + self.brain.Theta[7]  # output of n8

        activation_left = self.brain.sigmoid(o7 * self.MW[0] + o8 * self.MW[2])
        activation_right = self.brain.sigmoid(o7 * self.MW[1] + o8 * self.MW[3])

        # Update timer:
        if self.timer_motor_l > 0:
            self.timer_motor_l -= self.brain.step_size
        if self.timer_motor_r > 0:
            self.timer_motor_r -= self.brain.step_size

        # We set timer to 0.5. That means we have max. 2 clicks per time-unit
        if activation_left > threshold:
            if self.timer_motor_l <= 0:
                self.timer_motor_l = 0.5  # reset the timer
                activation[0] = -1  # set left activation to -1 to influence velocity to the left

        if activation_right > threshold:
            if self.timer_motor_r <= 0:
                self.timer_motor_r = 0.5
                activation[1] = 1  # set right to one to influence velocity to the right

        return activation

    @staticmethod
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

    @staticmethod
    def add_noise(state):
        magnitude = np.random.normal(0, 0.05)
        return state + magnitude


class EmbodiedAgentV1(Agent):
    """
    This is a version of the agent that is more embodied.
    The agent receives positions of target and environment borders in terms of their distance to the position
    of the tracker (which implements the agent's "body"). The agent also has a different anatomy with respect to
    the non-embodied agent, see network2.pdf for a picture.
    """
    def __init__(self, network, agent_parameters, screen_width):
        # change visual input: 3 distance sensors for border_left, border_right, target
        # each sensor connected with one connection to 3 different neurons (1, 2, 3)
        agent_parameters["n_visual_sensors"] = 3
        agent_parameters["n_visual_connections"] = 1
        agent_parameters["n_audio_sensors"] = 2
        agent_parameters["n_audio_connections"] = 1
        agent_parameters["n_effector_connections"] = 2

        Agent.__init__(self, network, agent_parameters)
        self.screen_width = screen_width

    def visual_input(self, position_tracker, position_target):
        """
        The visual input to the agent
        :param position_tracker: absolute position of the tracker 
        :param position_target: absolute position of the target
        :return:
        """
        # distance to borders is absolute
        dleft_border = self.add_noise(abs(self.screen_width[0] - position_tracker))
        dright_border = self.add_noise(abs(self.screen_width[1] - position_tracker))
        # distance to target is not: neg when target is on the left, pos when on the right
        dtarget = self.add_noise(position_target - position_tracker)

        self.brain.I[0] = self.VW[0] * dtarget  # to n1
        self.brain.I[1] = self.VW[1] * dleft_border  # to n2
        self.brain.I[2] = self.VW[2] * dright_border  # to n3

    def auditory_input(self, sound_input):
        """
        The auditory input to the agent
        :param sound_input: Tone(s) induced by left and/or right click
        """
        left_click, right_click = sound_input[0], sound_input[1]

        self.brain.I[3] = self.AW[0] * left_click  # to n4
        self.brain.I[5] = self.AW[1] * right_click  # to n6

    def motor_output(self):
        """
        The motor output of the agent
        :return: output
        """
        # Set activation threshold
        activation = [0, 0]  # Initial activation is zero
        # threshold = 0  # Threshold for output
        threshold = 0.5  # Threshold for output

        # consider adding noise to output before multiplying by motor gains,
        # drawn from a Gaussian distribution with (mu=0, var=0.05)
        o7 = self.brain.Y[6] + self.brain.Theta[6]  # output of n7
        o8 = self.brain.Y[7] + self.brain.Theta[7]  # output of n8

        # activation_left = self.brain.sigmoid(o7 * self.MW[0])
        # activation_right = self.brain.sigmoid(o8 * self.MW[1])
        activation_left = self.brain.sigmoid(o7 * self.MW[0] + o8 * self.MW[2])
        activation_right = self.brain.sigmoid(o7 * self.MW[1] + o8 * self.MW[3])

        # Update timer:
        if self.timer_motor_l > 0:
            self.timer_motor_l -= self.brain.step_size
        if self.timer_motor_r > 0:
            self.timer_motor_r -= self.brain.step_size

        # We set timer to 0.5. That means we have max. 2 clicks per time-unit
        # Version by GK
        if activation_left >= threshold:
            if activation_right >= threshold:  # both over threshold: choose one in proportion to relative activation
                if random.random() <= activation_left / (activation_left + activation_right):
                    if self.timer_motor_l <= 0:  # left wins
                        self.timer_motor_l = 0.5
                        activation[0] = -1
                elif self.timer_motor_r <= 0:  # right wins
                    self.timer_motor_r = 0.5
                    activation[1] = 1
            elif self.timer_motor_l <= 0:  # only left is over threshold
                self.timer_motor_l = 0.5
                activation[0] = -1
        elif activation_right >= threshold:
            if self.timer_motor_r <= 0:
                self.timer_motor_r = 0.5
                activation[1] = 1

        return activation


class EmbodiedAgentV2(Agent):
    """
    This is a version of the agent that is more embodied.
    The agent receives positions of target and environment borders in terms of their distance to the position
    of the tracker (which implements the agent's "body"). Compared to V1 its perception is in terms of separate
    distance to the target on the left or right and inverted visual stimulation (the smaller the distance,
    the larger the stimulation; linearly scaled to be max 10).
    See network3.pdf for a picture.
    """
    def __init__(self, network, agent_parameters, screen_width):
        # change visual input: 4 absolute distance sensors for border_left, border_right, target_left, target_right
        # each sensor connected with 1 connection to 4 different neurons (1, 2, 3, 4)
        # each auditory sensor with 1 connection to 2 different neurons (5, 6)
        # each motor with 1 connection to 2 different neurons (7, 8)
        agent_parameters["n_visual_sensors"] = 4
        agent_parameters["n_visual_connections"] = 1
        agent_parameters["n_audio_sensors"] = 2
        agent_parameters["n_audio_connections"] = 1
        agent_parameters["n_effector_connections"] = 2

        Agent.__init__(self, network, agent_parameters)
        self.screen_width = screen_width
        self.max_dist = self.screen_width[1] - self.screen_width[0]
        self.visual_scale = self.max_dist / agent_parameters["max_visual_activation"]

    def visual_input(self, position_tracker, position_target):
        """
        The visual input to the agent
        :param position_tracker: absolute position of the tracker 
        :param position_target: absolute position of the target
        :return:
        """
        dleft_border = self.add_noise((self.max_dist-abs(self.screen_width[0]-position_tracker))/self.visual_scale)
        dright_border = self.add_noise((self.max_dist-abs(self.screen_width[1]-position_tracker))/self.visual_scale)

        if position_target > position_tracker:
            # target is to the right of the tracker
            dleft_target = 0
            dright_target = self.add_noise((self.max_dist-abs(position_target-position_tracker))/self.visual_scale)
        elif position_target < position_tracker:
            # target is to the left of the tracker
            dleft_target = self.add_noise((self.max_dist-abs(position_target-position_tracker))/self.visual_scale)
            dright_target = 0
        else:
            # if tracker is on top of the target, both eyes are activated to the maximum
            dleft_target = self.add_noise((self.max_dist-abs(position_target-position_tracker))/self.visual_scale)
            dright_target = self.add_noise((self.max_dist-abs(position_target-position_tracker))/self.visual_scale)

        self.brain.I[0] = self.VW[0] * dleft_border  # to n1
        self.brain.I[1] = self.VW[1] * dright_border  # to n2
        self.brain.I[2] = self.VW[2] * dleft_target  # to n3
        self.brain.I[3] = self.VW[3] * dright_target  # to n4

    def auditory_input(self, sound_input):
        """
        The auditory input to the agent
        :param sound_input: Tone(s) induced by left and/or right click
        """
        left_click, right_click = sound_input[0], sound_input[1]

        self.brain.I[4] = self.AW[0] * left_click  # to n5
        self.brain.I[5] = self.AW[1] * right_click  # to n6

    def motor_output(self):
        """
        The motor output of the agent
        :return: output
        """
        # Set activation threshold
        activation = [0, 0]  # Initial activation is zero
        # threshold = 0  # Threshold for output
        threshold = 0.5  # Threshold for output

        # consider adding noise to output before multiplying by motor gains,
        # drawn from a Gaussian distribution with (mu=0, var=0.05)
        o7 = self.brain.Y[6] + self.brain.Theta[6]  # output of n7
        o8 = self.brain.Y[7] + self.brain.Theta[7]  # output of n8

        # activation_left = self.brain.sigmoid(o7 * self.MW[0])
        # activation_right = self.brain.sigmoid(o8 * self.MW[1])
        activation_left = self.brain.sigmoid(o7 * self.MW[0] + o8 * self.MW[2])
        activation_right = self.brain.sigmoid(o7 * self.MW[1] + o8 * self.MW[3])

        # Update timer:
        if self.timer_motor_l > 0:
            self.timer_motor_l -= self.brain.step_size
        if self.timer_motor_r > 0:
            self.timer_motor_r -= self.brain.step_size

        # We set timer to 0.5. That means we have max. 2 clicks per time-unit
        # Version by GK
        if activation_left >= threshold:
            if activation_right >= threshold:  # both over threshold: choose one in proportion to relative activation
                if random.random() <= activation_left / (activation_left + activation_right):
                    if self.timer_motor_l <= 0:  # left wins
                        self.timer_motor_l = 0.5
                        activation[0] = -1
                elif self.timer_motor_r <= 0:  # right wins
                    self.timer_motor_r = 0.5
                    activation[1] = 1
            elif self.timer_motor_l <= 0:  # only left is over threshold
                self.timer_motor_l = 0.5
                activation[0] = -1
        elif activation_right >= threshold:
            if self.timer_motor_r <= 0:
                self.timer_motor_r = 0.5
                activation[1] = 1

        return activation


class ButtonOnOffAgent(EmbodiedAgentV2):
    """
    This is a version of the embodied agent (v2) with a modified button pressing mechanism.
    """
    def __init__(self, network, agent_parameters, screen_width):
        EmbodiedAgentV2.__init__(self, network, agent_parameters, screen_width)

    def motor_output(self):
        """
        If a button neuron's output (range [0, 1]) increases to more than or equal to 0.75,
        then its button is turned “on” and produces a “click.” The button is turned “off” when
        that neuron's output falls below 0.75.
        :return: output
        """
        # Set activation threshold
        activation = [0, 0]  # Initial activation is zero
        threshold = 0.75  # Threshold for output

        # consider adding noise to output before multiplying by motor gains,
        # drawn from a Gaussian distribution with (mu=0, var=0.05)
        o7 = self.brain.Y[6] + self.brain.Theta[6]  # output of n7
        o8 = self.brain.Y[7] + self.brain.Theta[7]  # output of n8
        activation_left = self.brain.sigmoid(o7 * self.MW[0])
        activation_right = self.brain.sigmoid(o8 * self.MW[1])

        if activation_left >= threshold and not self.button_state[0]:
            activation[0] = -1   # set left activation to -1 to influence velocity to the left
            self.button_state[0] = True
        elif activation_left < threshold and self.button_state[0]:
            self.button_state[0] = False

        if activation_right >= threshold and not self.button_state[1]:
            activation[1] = 1  # set right to one to influence velocity to the right
            self.button_state[1] = True
        elif activation_right < threshold and self.button_state[1]:
            self.button_state[1] = False

        # return activation, [activation_left, activation_right]
        return activation


class DirectVelocityAgent(EmbodiedAgentV2):
    """
    This is a version of the embodied agent (v2) with direct velocity control (no button pressing).
    It needs to be paired with a DirectTracker.
    """

    def __init__(self, network, agent_parameters, screen_width):
        EmbodiedAgentV2.__init__(self, network, agent_parameters, screen_width)

    def motor_output(self):
        """
        One neuron controls leftward, the other rightward velocity. Each velocity is calculated by mapping the activation
        in the range of [0, 1] to the range [-1, 1] and then multiplying by output gain.
        :return: output
        """
        # consider adding noise to output before multiplying by motor gains,
        # drawn from a Gaussian distribution with (mu=0, var=0.05)
        o7 = self.brain.sigmoid(self.brain.Y[6] + self.brain.Theta[6])  # output of n7
        o8 = self.brain.sigmoid(self.brain.Y[7] + self.brain.Theta[7])  # output of n8
        activation_left = self.linmap(o7, [0, 1], [-1, 1]) * self.MW[0]
        activation_right = self.linmap(o8, [0, 1], [-1, 1]) * self.MW[1]

        activation = [activation_left, activation_right]
        return activation

    # def motor_output(self):
    #     """
    #     One neuron controls leftward, the other rightward velocity. Each velocity is calculated by mapping the activation
    #     in the range of [0, 1] to the range [-1, 1] and then multiplying by output gain.
    #     :return: output
    #     """
    #     # consider adding noise to output before multiplying by motor gains,
    #     # drawn from a Gaussian distribution with (mu=0, var=0.05)
    #     o7 = self.brain.sigmoid(self.brain.Y[6] + self.brain.Theta[6])  # output of n7
    #     o8 = self.brain.sigmoid(self.brain.Y[7] + self.brain.Theta[7])  # output of n8
    #     activation_left = o7 * self.MW[0]
    #     activation_right = o8 * self.MW[1]
    #
    #     activation = [activation_left, activation_right]
    #     return activation

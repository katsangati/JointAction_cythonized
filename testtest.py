import analyze as az
import pstats, cProfile

import pyximport
pyximport.install()

import evolve
import simulate


config = az.load_config(None, None, None)
evolution = evolve.Evolution(config['evolution_params']['pop_size'],
                             config['evolution_params'],
                             config['network_params'],
                             config['evaluation_params'],
                             config['agent_params'])
pop = evolution.create_population(2)

# 10 min per generation of just experiment
simulation_run = simulate.Simulation(evolution.step_size, evolution.evaluation_params)
# cProfile.runctx("simulation_run.run_joint_trials(pop[0], pop[1], simulation_run.trials)", globals(), locals(), "Profile.prof")
cProfile.runctx("simulation_run.run_trials(pop[0], simulation_run.trials)", globals(), locals(), "Profile.prof")

# 0.007 s to reproduce 2 agents
#cProfile.runctx("evolution.reproduce(pop)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()

# from CTRNN import CTRNN
# cbrain = CTRNN(8, 0.01, [1, 100], [1, 1], [-15, 15], [-15, 15])
#
# print(cbrain.get_state())
# # cbrain.euler_step()
#
#
# cProfile.runctx("cbrain.euler_step()", globals(), locals(), "Profile.prof")
#
# s = pstats.Stats("Profile.prof")
# s.strip_dirs().sort_stats("time").print_stats()
#
# print(cbrain.get_state())



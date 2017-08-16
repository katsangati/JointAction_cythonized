import analyze as az

# random single agents
td2, ag2 = az.run_random_population(1, "all")

# check evolved agents
az.plot_fitness('single', 'buttons', '4264')
td1, ag1 = az.run_single_agent('single', 'buttons', '4264', 200, 0, "all")
az.check_generalization('single', 'buttons', '460496', ag1)

# additional checks
w = az.plot_weights('single', 'buttons', 123, [0, 10, 20, 30, 40], 1)
az.animate_trial('single', 'buttons', 123, 0, 0, 0)


# random joint agents
# td, a1, a2 = az.run_random_pair(3, 'all')

# check evolved joint agents
az.plot_fitness('joint', 'buttons', '921660')
td, a1, a2 = az.run_single_pair('buttons', '921660', 23, 0, 'all')




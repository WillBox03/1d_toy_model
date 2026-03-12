import numpy as np
import sys
import time

from ITV_engine import ITV_env
from optimiser import SequenceOptimiser

np.set_printoptions(threshold=sys.maxsize)

resolution = 0.01 # in cm

ctv_length = 2 # in cm

n_spots = 10

amp = 1 #in cm #

t_step = 0.1 # in s

region = 'itv'

env = ITV_env(resolution, ctv_length, n_spots, amp, t_step = t_step)

env.set_spot_weights('uniform', repaints = 9) # Uniform weighting

period = 5 # in s

starting_phase = None # in rads

phases = 18

weighting = False

lr_rast_dist, lr_rast_wmse = env.sim(period, 'lr_rast', starting_phase= starting_phase, phases = phases, region = region, weighting = weighting) 

# Right to left raster scan
rl_rast_dist, rl_rast_wmse = env.sim(period, 'rl_rast', starting_phase= starting_phase, phases = phases, region = region, weighting = weighting) 

# By leaving starting_phase as None , it automatically simulates all 8 phases.

# When averaging over all phases, lr and rl raster scans should be identical
lr_rast_dists, lr_rast_avg_wmse = env.sim(period, 'lr_rast')
rl_rast_dists, rl_rast_avg_wmse = env.sim(period, 'rl_rast')

# Max distances
max_time_seq = env.set_sequence('max_dist')
max_time_dist, max_time_wmse = env.sim(period, 'max_dist', starting_phase= starting_phase, phases = phases, region = region, weighting = weighting)
max_dists, max_avg_wmse = env.sim(period, 'max_dist')
print(f'The maximum time sequence is: {max_time_seq}')
# Random sequences requires saving of the random sequence to be reused
random_sequence = env.set_sequence('rand')
print(f"The random sequence is: {random_sequence}")

# Pass in as an explicitly saved array into the sim model
rand_dist, rand_wmse = env.sim(period, random_sequence, starting_phase= starting_phase, phases = phases, region = region, weighting = weighting)
rand_dists, rand_avg_wmse = env.sim(period, random_sequence)

# Find the optimal sequence
start_time = time.perf_counter()
env.calculate_mask_tensor(period, starting_phase = starting_phase, phases = phases, region = region)
end_time = time.perf_counter()
duration = end_time - start_time
print(f"Tensor mask loaded in {duration:.4f} seconds")

optimiser = SequenceOptimiser(env)

optimal_sequence, optimal_wmse, _ = optimiser.run('simanneal', iterations = 100000, pop_size = 50, weighting = weighting)
optimal_dist, _ = env.sim(period, optimal_sequence, starting_phase= starting_phase, phases = phases, region = region, weighting = weighting)
print(f'Optimal sequence (simulated annealing) is {optimal_sequence}')

sims = [lr_rast_dist, max_time_dist, optimal_dist]
names = ['Left-Right Raster', 'Max Time', 'Optimised']

print(f"""wmses for a starting phase of 0 are:
      {names[0]}:{lr_rast_wmse}
      {names[1]}:{max_time_wmse}
      {names[2]}:{optimal_wmse}""")

'''print(f"""Average wmses for all starting phases:
      {names[0]}:{lr_rast_avg_wmse}
      {names[1]}:{rl_rast_avg_wmse}
      """)'''

# Display
env.display(sims, names, avg_phases = True)

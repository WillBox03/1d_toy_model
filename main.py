import numpy as np
import sys
import time

from ITV_engine import ITV_env
from optimiser import SequenceOptimiser

np.set_printoptions(threshold=sys.maxsize)

resolution = 0.01 # in cm

env_length = 2 # in cm

n_spots = 10

amp = 1 #in cm #

env = ITV_env(resolution, env_length, n_spots, amp)

env.set_spot_weights('uniform', repaints = 0) # Uniform weighting

t_step = 1/9 # in s

period = 2 # in s

starting_phase = 0 # in rads

lr_rast_dist, lr_rast_wmse = env.sim(t_step, period, 'lr_rast', starting_phase= starting_phase) 

# Right to left raster scan
rl_rast_dist, rl_rast_wmse = env.sim(t_step, period, 'rl_rast', starting_phase= starting_phase) 

# By leaving starting_phase as None , it automatically simulates all 8 phases.

# When averaging over all phases, lr and rl raster scans should be identical
lr_rast_dists, lr_rast_avg_wmse = env.sim(t_step, period, 'lr_rast')
rl_rast_dists, rl_rast_avg_wmse = env.sim(t_step, period, 'rl_rast')

# Max distances
max_time_dist, max_time_wmse = env.sim(t_step, period, 'max_dist', starting_phase= starting_phase)
max_dists, max_avg_wmse = env.sim(t_step, period, 'max_dist')

# Random sequences requires saving of the random sequence to be reused
random_sequence = env.set_sequence('rand')
print(f"The random sequence is: {random_sequence}")

# Pass in as an explicitly saved array into the sim model
rand_dist, rand_wmse = env.sim(t_step, period, random_sequence, starting_phase= starting_phase)
rand_dists, rand_avg_wmse = env.sim(t_step, period, random_sequence)

# Find the optimal sequence
start_time = time.perf_counter()
env.calculate_mask_tensor(t_step, period, starting_phase = starting_phase)
end_time = time.perf_counter()
duration = end_time - start_time
print(f"Tensor mask loaded in {duration:.4f} seconds")

optimiser = SequenceOptimiser(env)

optimal_sequence, optimal_wmse_optimiser = optimiser.run('exhaustive')
optimal_dist, _ = env.sim(t_step, period, optimal_sequence, starting_phase=0.0)
print(f'Optimal sequence (exhaustive search) is {optimal_sequence}')

sims = [lr_rast_dist, rl_rast_dist, optimal_dist]
names = ['Left-to-Right Raster', 'Right-to-Left Raster', 'Optimal Sequence']

print(f"""wmses for a starting phase of 0 are:
      {names[0]}:{lr_rast_wmse}
      {names[1]}:{rl_rast_wmse}
      {names[2]}:{optimal_wmse_optimiser}""")

'''print(f"""Average wmses for all starting phases:
      {names[0]}:{lr_rast_avg_wmse}
      {names[1]}:{rl_rast_avg_wmse}
      """)'''

# Display
env.display_sims(sims, names)

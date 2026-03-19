import numpy as np
import sys
import time

from ITV_engine import ITV_env_1D
from optimiser import SequenceOptimiser

np.set_printoptions(threshold=sys.maxsize)

resolution = 0.01 # in cm

ctv_length = 2 # in cm

n_spots = 10

spot_size = 0.4

amp = 1 #in cm

t_step = 0.05 # in s

region = 'itv'

env = ITV_env_1D(resolution, ctv_length, amp, n_spots = n_spots, spot_size = spot_size, t_step = t_step)

env.set_spot_weights('uniform', repaints = 19) # Uniform weighting

period = 5 # in s

starting_phase = 0 # in rads

phases = 24

weighting = False

motion = 'sin6'

lr_rast_dist, lr_rast_mse = env.sim(period, 'lr_rast', starting_phase= starting_phase, n_phases = phases, region = region, weighting = weighting, motion = motion) 

# Right to left raster scan
rl_rast_dist, rl_rast_mse = env.sim(period, 'rl_rast', starting_phase= starting_phase, n_phases = phases, region = region, weighting = weighting, motion = motion) 

# Max distances
max_time_seq = env.set_sequence('max_dist')
max_time_dist, max_time_mse = env.sim(period, 'max_dist', starting_phase= starting_phase, n_phases = phases, region = region, weighting = weighting, motion = motion)
max_dists, max_avg_mse = env.sim(period, 'max_dist')
print(f'The maximum time sequence is: {max_time_seq}')
# Random sequences requires saving of the random sequence to be reused
random_sequence = env.set_sequence('rand')
print(f"The random sequence is: {random_sequence}")

# Pass in as an explicitly saved array into the sim model
rand_dist, rand_mse = env.sim(period, random_sequence, starting_phase= starting_phase, n_phases = phases, region = region, weighting = weighting, motion = motion)

# Find the optimal sequence
start_time = time.perf_counter()
env.calculate_mask_tensor(period, starting_phase = starting_phase, n_phases = phases, region = region, motion = motion)
end_time = time.perf_counter()
duration = end_time - start_time
print(f"Tensor mask loaded in {duration:.4f} seconds")

optimiser = SequenceOptimiser(env)

optimal_sequence, optimal_mse, _ = optimiser.run('simanneal', iterations = 20000, pop_size = 20, weighting = weighting)
optimal_dist, _ = env.sim(period, optimal_sequence, starting_phase= starting_phase, n_phases = phases, region = region, weighting = weighting, motion = motion)
print(f'Optimal sequence (simulated annealing) is {optimal_sequence}')

sims = [lr_rast_dist, rand_dist, max_time_dist, optimal_dist]
names = ['Left-Right Raster', 'Random', 'Max-Time', 'Sim Anneal']

print(f"""wmses for a starting phase of {starting_phase} are:
      {names[0]}:{lr_rast_mse},
      {names[1]}:{rand_mse},
      {names[2]}:{max_time_mse},
      {names[3]}:{optimal_mse}
      """)

doses = {
      'Standard Raster': lr_rast_dist,
      'Optimal': optimal_dist
}

# Display
env.display(sims, labels= names, avg_phases=True)
env.display_dvh(doses, regions=['itv'])
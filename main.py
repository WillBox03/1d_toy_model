import ITV_engine
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

resolution = 0.01 # in cm

ctv_length = 2 # in cm

n_spots = 8

amp = 1 #in cm 
 
ctv = ITV_engine.ITV_env(resolution, ctv_length, n_spots, amp)

ctv.set_spot_weights(np.ones(n_spots)) # Uniform weighting

t_step = 0.2 # in s

period = 7 # in s

'''lr_rast_dist = ctv.sim(t_step, freq, 'lr_rast', 3/2*np.pi) # Left to right raster scan
lr_rast_mse = ctv.calc_mse(lr_rast_dist)
print(lr_rast_mse)

rl_rast_dist = ctv.sim(t_step, freq, 'rl_rast', 3/2*np.pi) # Right to left raster scan
rl_rast_mse = ctv.calc_mse(rl_rast_dist)
print(lr_rast_mse)

# When averaging over all phases, lr and rl raster scans should be identical
#lr_rast_dists, lr_rast_avg_mse = ctv.sim_phases(t_step, freq, 'lr_rast')
#rl_rast_dists, rl_rast_avg_mse = ctv.sim_phases(t_step, freq, 'rl_rast')

#rand_dist = ctv.sim(t_step, freq, 'rand', 0)
#rand_dists, rand_avg_mse = ctv.sim_phases(t_step, freq, 'rand') # Random sequence scan'''

#max_time_dist = ctv.sim(t_step, period, 'max_dist,0)

min_mse_dist = ctv.sim(t_step, period, np.array([6, 1, 4, 5, 0, 7, 2, 3]), 0)
min_mse_dist_mse = ctv.calc_mse(min_mse_dist)
print(f"Minimum distribution using mse as evaluation gives mse of {min_mse_dist_mse} ")
min_mse_dist_wmse = ctv.calc_wmse(min_mse_dist)
print(f"Minimum distribution using mse as evaluation gives wmse of {min_mse_dist_wmse} ")

min_wmse_dist = ctv.sim(t_step, period, np.array([6, 1, 4, 5, 0, 7, 2, 3]), 0)
min_wmse_dist_mse = ctv.calc_mse(min_wmse_dist)
print(f"Minimum distribution using wmse as evaluation gives mse of {min_mse_dist_mse} ")
min_wmse_dist_wmse = ctv.calc_wmse(min_wmse_dist)
print(f"Minimum distribution using wmse as evaluation gives wmse of {min_wmse_dist_wmse} ")

sims = np.array([min_mse_dist])

ctv.display_sims(sims)

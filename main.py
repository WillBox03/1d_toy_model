import ITV_engine
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

resolution = 0.01 # in cm

ctv_length = 2 # in cm

n_spots = 10

amp = 1 #in cm 
 
ctv = ITV_engine.ITV_env(resolution, ctv_length, n_spots, amp)

ctv.set_spot_weights(np.ones(n_spots)) # Uniform weighting

t_step = 0.1 # in s

period = 5 # in s

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

min_dist = ctv.sim(t_step, period, np.array([9, 4, 1, 6, 0, 8, 5, 7, 2, 3]), 0)
min_dist_mse = ctv.calc_mse(min_dist)
print(min_dist_mse)
#print(lr_rast_avg_mse, rl_rast_avg_mse, rand_avg_mse)

sims = np.array([min_dist])

ctv.display_sims(sims)

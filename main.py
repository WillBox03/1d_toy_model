import PTV_engine
import numpy as np

resolution = 0.01 # in cm

ptv_length = 2 # in cm

n_spots = 10

amp = 0.5 #in cm 
 
ptv = PTV_engine.PTV_env(resolution, ptv_length, n_spots, amp)

ptv.set_spot_weights(np.ones(n_spots))

t_step = 1/9 # in seconds

freq = 1/2 # in hertz

lr_rast_dist = ptv.sim(t_step, freq, 'lr_rast', 0) # Left to right raster scan

lr_rast_mse = ptv.calc_mse(lr_rast_dist)
print(lr_rast_mse)
lr_rast_dists = ptv.sim_phases(t_step, freq, 'lr_rast')
rl_rast_dist = ptv.sim(t_step, freq, 'rl_rast', np.pi/2) # Right to left raster scan

rand_dist = ptv.sim(t_step, freq, 'rand', 0) # Random sequence scan

max_time_dist = ptv.sim(t_step, freq, np.array([5,0,9,1,8,2,7,3,6,4]), 0)

sims = np.array([lr_rast_dist])
sims = np.array([lr_rast_dists[0]])

ptv.display_sims(sims)

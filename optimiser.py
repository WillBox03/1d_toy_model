import ITV_engine
import numpy as np
from math import factorial
import time

import sys

np.set_printoptions(threshold=sys.maxsize)

resolution = 0.01 # in cm

ctv_length = 2 # in cm

n_spots = 8

amp = 1 #in cm 

t_step = 0.1

period = 2

env = ITV_engine.ITV_env(resolution, ctv_length, n_spots, amp)

env.set_spot_weights(np.ones(n_spots))

start_time = time.perf_counter()
env.calculate_mask_tensor(t_step, period)
end_time = time.perf_counter()
duration = end_time - start_time

print(f"Task completed in {duration:.4f} seconds")


def optimiser_sim(env, type, n_samples = None):
    '''
    Uses various methods to optimise the spot sequence by minimising the MSE of the distribution
    Args:
        env: An instance of the INV_env class with a precomputed tensor
        n_samples: The number of samples required for the simulation
        type (string): The type of optimiser algorithm used, currently accepts "montecarlo" and "exhaustive"
    '''
    n_spots = env.n_spots
    
    if type == "montecarlo":
        #Create numpy random number generator:
        rng = np.random.default_rng()
        
        start_time = time.perf_counter()
        #Generate random sequences, shuffling each row
        sequences = rng.permuted(np.tile(np.arange(n_spots), (n_samples, 1)), axis=1)

        #Evaluate mses of each sequence
        mses = env.evaluate_sequences(sequences)
    
        #Fnd the lowest mse index
        best_idx = np.argmin(mses)
        
        #Calculates computation time
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Task completed in {duration:.4f} seconds")
    
        return sequences[best_idx], mses[best_idx]
    
    elif type == "exhaustive":
        
        start_time = time.perf_counter()

        #Efficient method to generate all permutations
        sequences = np.zeros((factorial(n_spots), n_spots), np.int16)
        fact = 1
        for m in range(2, n_spots+1):
            frame = sequences[:fact, n_spots-m+1:]
            for i in range(1, m):
                sequences[i*fact:(i+1)*fact, n_spots-m] = i
                sequences[i*fact:(i+1)*fact, n_spots-m+1:] = frame + (frame >= i)
            frame += 1
            fact *= m
        
        mses = env.evaluate_sequences(sequences)
    
        #Fnd the lowest mse index
        best_idx = np.argmin(mses)
        
        #Calculates computation time
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Task completed in {duration:.4f} seconds")
        return sequences[best_idx], mses[best_idx]

print(optimiser_sim(env, "montecarlo", 10000))
print(optimiser_sim(env, "exhaustive"))

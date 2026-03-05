import numpy as np
from optimiser import SequenceOptimiser
from ITV_engine import ITV_env
import matplotlib.pyplot as plt
import time
import gc

if __name__ == "__main__":

    resolution = 0.01 # in cm

    env_length = 2 # in cm

    n_spots = 10

    amp = 1 #in cm 

    t_step = 0.1

    period = 5

    env = ITV_env(resolution, env_length, n_spots, amp, t_step = t_step)

    env.set_spot_weights('uniform', repaints = 10)

    start_time = time.perf_counter()
    env.calculate_mask_tensor(period)
    end_time = time.perf_counter()
    duration = end_time - start_time

    print(f"Tensor mask loaded in {duration:.4f} seconds")
    
    opt = SequenceOptimiser(env)
    
    repeats = 5
    
    baseline_raster_sequence = env.set_sequence('lr_rast')
    baseline_raster = env.evaluate_sequences(baseline_raster_sequence)
    #sequence_ex, mse_ex = opt.run("exhaustive", weighting = True) 
    print(baseline_raster_sequence)
    print(baseline_raster)
    times = 4
    
    mse_sa = np.zeros((times, repeats, 2))
    mse_mc = np.zeros((times, repeats, 2))
    mse_h  = np.zeros((times, repeats, 2))
    
    sa_params = np.array([1100,5600,12000,28000])
    mc_params = np.array([3500, 18000, 360000, 900000])
    h_params = np.array([[250,4],[900,10],[900,20],[1500,35]])
    
    print("Warming up compilers...")
    opt.run('simanneal', iterations=10, pop_size=2, temp=0.3, final_temp=0.001)
    opt.run('montecarlo', n_samples=10)
    opt.run('mcghybrid')
    
    for j in range(times):
        for i in range(repeats):
        
            gc.collect()
            
            opt = SequenceOptimiser(env)
            
            _, mse_sa[j,i,0], mse_sa[j,i,1] = opt.run('simanneal', iterations = sa_params[j], pop_size = 20, temp = 0.3, final_temp = 0.001)
            
            _, mse_mc[j,i,0] ,mse_mc[j,i,1] = opt.run("montecarlo",n_samples = mc_params[j])
            
            _, mse_h[j,i,0], mse_h[j,i,1] = opt.run("mcghybrid", n_samples = h_params[j,0], generations = h_params[j,1])
            
            time.sleep(0.2 * (j + 1))

    sa_means = np.mean(mse_sa, axis=1) # Shape: (5, 2)
    sa_stds  = np.std(mse_sa, axis=1)

    # Monte Carlo Stats
    mc_means = np.mean(mse_mc, axis=1)
    mc_stds  = np.std(mse_mc, axis=1)

    # Hybrid Stats
    h_means  = np.mean(mse_h, axis=1)
    h_stds   = np.std(mse_h, axis=1)
    
    plt.figure(figsize=(10, 6))

    # Plot SA (Blue)
    plt.errorbar(
        x=sa_means[:, 1],      # X = Average Time
        y=sa_means[:, 0],      # Y = Average MSE
        xerr=sa_stds[:, 1],    # Horizontal Error Bars (Time Variance)
        yerr=sa_stds[:, 0],    # Vertical Error Bars (MSE Variance)
        label='Simulated Annealing',
        fmt='-o', capsize=5
    )

    # Plot MC (Orange)
    plt.errorbar(
        x=mc_means[:, 1], 
        y=mc_means[:, 0], 
        xerr=mc_stds[:, 1], 
        yerr=mc_stds[:, 0], 
        label='Monte Carlo',
        fmt='-s', capsize=5
    )

    # Plot Hybrid (Green)
    plt.errorbar(
        x=h_means[:, 1], 
        y=h_means[:, 0], 
        xerr=h_stds[:, 1], 
        yerr=h_stds[:, 0], 
        label='Hybrid',
        fmt='-^', capsize=5
    )
    
    plt.axhline(y=baseline_raster, color='red', linestyle='--', linewidth=2, label='Baseline (Raster)')
    #plt.axhline(y=mse_ex, color='green', linestyle='--', linewidth=2, label='Optimal (Exhaustive)')
    
    plt.title('Time-Accuracy Trade-off')
    plt.xlabel('Execution Time (s)')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
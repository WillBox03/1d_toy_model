import numpy as np
from optimiser import SequenceOptimiser
from ITV_engine import ITV_env
import matplotlib.pyplot as plt
import gc

if __name__ == "__main__":

    resolution = 0.01 # in cm

    env_length = 2 # in cm

    n_spots = 10

    amp = 1 # in cm 

    t_step = np.linspace(0.01, 2, 50) # in seconds
    period = np.linspace(0.01, 2, 50) # in hertz

    env = ITV_env(resolution, env_length, n_spots, amp)

    env.set_spot_weights(np.ones(n_spots))
    
    opt = SequenceOptimiser(env)
    
    raster = np.arange(n_spots)
    
    repeats = 5
    
    improvement_times = np.zeros((len(t_step),len(period),repeats))
    
    param_vary_dists_lr = np.zeros((50, 50))

    for i in range(len(t_step)):
        for j in range(len(period)):
            param_vary_dists_lr[i, j] = env.sim_phases(t_step[i], period[j], 'lr_rast')[1]
    
    for i in range(len(t_step)):
        for j in range(len(period)):
            
            env.calculate_mask_tensor(t_step[i], period[j])
            
            for k in range(repeats):
                gc.collect()
                opt = SequenceOptimiser(env)
                _, _, improvement_times[i,j,k], _ = opt.run("mcghybrid", n_samples = 1000, generations = 50, time_track = True, comparison_sequence = raster, target_improvement = True)

    avg_improvement_times = np.mean(improvement_times, axis=2)
    
    fig, axes = plt.subplots(1, 2, figsize=(18,5))

    extent = [period.min(), period.max(), t_step.min(), t_step.max()]

    #Raster MSE heatmap
    im0 = axes[0].imshow(param_vary_dists_lr, aspect='auto', origin='lower', extent=extent, cmap='viridis')
    axes[0].set_title("Baseline Raster MSE")
    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar0.set_label("MSE")
    
    im1 = axes[1].imshow(avg_improvement_times, aspect='auto', origin='lower', extent=extent, cmap='viridis')
    axes[1].set_title("Hybrid Search Sequence MSE")
    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.set_label("MSE")

    # Add the colorbar
    cbar = plt.colorbar(axes[1])
    cbar.set_label("Time to Beat Raster Baseline (Seconds)")

    # Label the axes
    plt.xlabel("Breathing Period (Seconds)")
    plt.ylabel("Beam Movement Time Step (Seconds)")
    plt.title("Algorithm Efficiency Across Motion Parameters")

    plt.tight_layout()
    plt.show()

    
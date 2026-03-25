import numpy as np
import matplotlib.pyplot as plt
import time
import gc
import os

from optimiser import SequenceOptimiser
from ITV_engine import ITV_env_1D

def autotune_optimisers(opt, target_times = [1.0, 2.0, 5.0, 10.0]):
    """
    Runs micro-benchmarks to callibrate hyperparameter scalings
    """
    print('Autotuning Hyperparameters...')
    
    # Calibrating MC
    base_samples = 1000000
    start = time.perf_counter()
    opt.run('montecarlo', n_samples = base_samples)
    mc_test_time = time.perf_counter() - start
    mc_samples_per_sec = base_samples / mc_test_time
    mc_params = np.array([int(t * mc_samples_per_sec) for t in target_times], dtype=int)
    print(f'MC speed: {mc_samples_per_sec:,.0f} samples/sec')
    
    # Calibrating hybrid
    base_h_pop = 10
    base_h_samples = 500
    base_h_gens = 5
    
    start = time.perf_counter()
    opt.run('mcghybrid', n_samples=base_h_samples, generations=base_h_gens, population_size=base_h_pop)
    h_sprint_time = time.perf_counter() - start
    
    time_per_gen = h_sprint_time / base_h_gens
    time_per_individual = time_per_gen / base_h_pop
    
    needs_fallback = (target_times[0] / time_per_gen) < 1.0 # Check if shortest target time is too fast for baseline
    
    h_params = []
    for i, t in enumerate(target_times):
        # Fallback logic if needed
        if needs_fallback:
            # Start at 5, and increase by 1 for each time step
            target_pop = 5 + i 
            target_samples = target_pop * 50 # Keep MC samples proportional to population size
            
            # Calculate exactly how long this smaller generation will take
            adjusted_time_per_gen = time_per_individual * target_pop
            target_gens = max(1, int(t / adjusted_time_per_gen))
            
        else:
            target_pop = base_h_pop
            target_samples = base_h_samples + (100 * i)
            target_gens = max(1, int(t / time_per_gen))
        
        h_params.append([target_samples, target_gens, target_pop])
    
    h_params = np.array(h_params, dtype=int)
    print(f"HYB speed: {time_per_gen:.3f} sec/gen")
    
    # Calibrate SA
    base_sa_pop = 20
    base_sa_iters = 100000
    start = time.perf_counter()
    opt.run('simanneal', iterations=base_sa_iters, pop_size=base_sa_pop, temp=0.3, final_temp=0.001)
    sa_sprint_time = time.perf_counter() - start
    
    sa_evals_per_sec = (base_sa_iters * base_sa_pop) / sa_sprint_time
    sa_params = []
    for i, t in enumerate(target_times):
        target_pop = base_sa_pop + (i * 10) 
        total_evals = t * sa_evals_per_sec
        target_iters = int(total_evals / target_pop)
        sa_params.append([target_pop, target_iters])
        
    sa_params = np.array(sa_params, dtype=int)
    print(f"SA speed:  {sa_evals_per_sec:,.0f} total evals/sec")
    
    return mc_params, h_params, sa_params
    

def optimiser_analysis(ctv_length, amp, period, t_step=0.1, n_spots=10, repaints=0, starting_phase=None, resolution=0.01, spot_size=None, repeats=5, target_times = [1.0, 2.0, 5.0, 10.0], save_directory="./benchmarks"):
    """
    Runs sequence optimisation benchmarking tests for a given environment
    and saves the output graph, with optimal dose distributions plotted
    """
    
    # Ensures save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Format filename
    phase_str = "None" if starting_phase is None else f"{starting_phase:.2f}"
    spot_size_str = "Auto" if spot_size is None else f"{spot_size:.2f}"
    filename = (f"ctv{ctv_length}_amp{amp}_spots{n_spots}_sz{spot_size_str}_"
                f"per{period}_step{t_step}_rep{repaints}_ph{phase_str}.png")
    filepath = os.path.join(save_directory, filename)
    
    print("Starting Benchmark...")   
    
    # Initialise environment
    env = ITV_env_1D(resolution, ctv_length, n_spots, amp, t_step = t_step)
    env.set_spot_weights('uniform', repaints = repaints)

    start_time = time.perf_counter()
    env.calculate_mask_tensor(period, starting_phase)

    print(f"Tensor mask loaded in {time.perf_counter() - start_time:.4f} seconds")
    
     # Get baseline raster
    baseline_lr_seq = env.set_sequence('lr_rast')
    baseline_lr_mse= env.evaluate_sequences(baseline_lr_seq)[0]
    baseline_rl_seq = env.set_sequence('rl_rast')
    baseline_rl_mse = env.evaluate_sequences(baseline_rl_seq)[0]
    
    # Initialise Optimiser
    opt = SequenceOptimiser(env)
    
    # Find global optimal sequence for small sequence length environments
    total_sequence_length = len(baseline_lr_seq)
    run_exhaustive = total_sequence_length <= 10
    
    if run_exhaustive:
        print(f"Sequence length is {total_sequence_length}. Running Exhaustive Search...")
        seq_ex, mse_ex = opt.run("exhaustive", weighting=True)
        
        # Simulate the physical dose for the exhaustive sequence
        ex_dose, _ = env.sim(period, seq_ex, starting_phase)
        ex_avg_dose = np.mean(ex_dose, axis=0) if ex_dose.ndim > 1 else ex_dose

    # Preallocate arrays
    times = len(target_times)
    mse_mc = np.zeros((times, repeats, 2))
    mse_h  = np.zeros((times, repeats, 2))
    mse_sa = np.zeros((times, repeats, 2))
    
    # Warm up compliers for more accurate time benchmarks
    print("Warming up compilers...")
    opt.run('simanneal', iterations=100, pop_size=2, temp=0.3, final_temp=0.001)
    opt.run('montecarlo', n_samples=100)
    opt.run('mcghybrid', n_samples=100, generations=2, population_size=5)

    
    mc_params, h_params, sa_params = autotune_optimisers(opt, target_times)

    global_best_mse = np.inf
    global_best_seq = None
    
    # Main benchmarking loop
    print('Running optimisations...')
    for j in range(times):
        for i in range(repeats):
        
            gc.collect()
            opt = SequenceOptimiser(env)
            
            # Monte Carlo
            seq_mc, mse_mc[j,i,0] ,mse_mc[j,i,1] = opt.run("montecarlo",n_samples = mc_params[j])
            if mse_mc[j,i,0] < global_best_mse:
                global_best_mse = mse_mc[j,i,0]
                global_best_seq = seq_mc.copy()
            
            # Hybrid
            seq_h, mse_h[j,i,0], mse_h[j,i,1] = opt.run("mcghybrid", n_samples = h_params[j,0], generations = h_params[j,1], population_size = h_params[j,2])
            if mse_h[j,i,0] < global_best_mse:
                global_best_mse = mse_h[j,i,0]
                global_best_seq = seq_h.copy()
                
            # Simulated Annealing
            seq_sa, mse_sa[j,i,0], mse_sa[j,i,1] = opt.run('simanneal', iterations = sa_params[j, 1], pop_size = sa_params[j, 0], temp = 0.3, final_temp = 0.001)
            if mse_sa[j,i,0] < global_best_mse:
                global_best_mse = mse_sa[j,i,0]
                global_best_seq = seq_sa.copy()
                
            time.sleep(0.25 * (j + 1)) # Help avoid CPU throttling

    # Calculate plot statistics
    mc_means = np.mean(mse_mc, axis=1)
    mc_stds  = np.std(mse_mc, axis=1)
    h_means  = np.mean(mse_h, axis=1)
    h_stds   = np.std(mse_h, axis=1)
    sa_means = np.mean(mse_sa, axis=1)
    sa_stds  = np.std(mse_sa, axis=1)
    
    # Physical dose profiles
    # Average only if multiple phases were tested
    lr_dose, _ = env.sim(period, baseline_lr_seq, starting_phase)
    lr_avg_dose = np.mean(lr_dose, axis=0) if lr_dose.ndim > 1 else lr_dose
    
    rl_dose, _ = env.sim(period, baseline_rl_seq, starting_phase)
    rl_avg_dose = np.mean(rl_dose, axis=0) if rl_dose.ndim > 1 else rl_dose
    
    opt_dose, _ = env.sim(period, global_best_seq, starting_phase)
    opt_avg_dose = np.mean(opt_dose, axis = 0) if opt_dose.ndim > 1 else opt_dose
    
    intended_dose = env.get_nominal_dist()
    
    # Create Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot MC
    ax1.errorbar(x=mc_means[:, 1],  y=mc_means[:, 0],  xerr=mc_stds[:, 1],  yerr=mc_stds[:, 0],  label='Monte Carlo', fmt='-s', capsize=5)

    # Plot Hybrid
    ax1.errorbar(x=h_means[:, 1],  y=h_means[:, 0],  xerr=h_stds[:, 1],  yerr=h_stds[:, 0],  label='Hybrid', fmt='-^', capsize=5)
    
    # Plot SA
    ax1.errorbar(x=sa_means[:, 1], y=sa_means[:, 0], xerr=sa_stds[:, 1], yerr=sa_stds[:, 0], label='Simulated Annealing', fmt='-o', capsize=5)
    
    # Plot raster baselines 
    if starting_phase is None:
        ax1.axhline(y=baseline_lr_mse.item(), color='red', linestyle='--', linewidth=2, label='Baseline Raster')
    else:
        ax1.axhline(y=baseline_lr_mse.item(), color='red', linestyle='--', linewidth=2, label='LR Raster')
        ax1.axhline(y=baseline_rl_mse.item(), color='darkred', linestyle=':', linewidth=2, label='RL Raster')
    
    ax1.set(title='Time-Accuracy Trade-off', xlabel='Execution Time (s)', ylabel='Average Error')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Y-axis padding
    max_mse = max(
        baseline_lr_mse.item(),
        baseline_rl_mse.item(),
        np.max(sa_means[:, 0] + sa_stds[:, 0]),
        np.max(mc_means[:, 0] + mc_stds[:, 0]),
        np.max(h_means[:, 0] + h_stds[:, 0])
    )
    ax1.set_ylim(top=max_mse * 1.10)
    
    # Isolate the best raster for plotting
    best_raster_mse = min(baseline_lr_mse.item(), baseline_rl_mse.item())
    if baseline_lr_mse.item() <= baseline_rl_mse.item():
        best_raster_dose = lr_avg_dose
        best_raster_label = f"Best Raster (LR) (MSE: {best_raster_mse:.3f})"
    else:
        best_raster_dose = rl_avg_dose
        best_raster_label = f"Best Raster (RL) (MSE: {best_raster_mse:.3f})"
        
    if run_exhaustive:
        ax1.axhline(y=mse_ex.item(), color='green', linestyle='-.', linewidth=2, label='Global Optimum')
        ax2.plot(env.env_space, ex_avg_dose, color='green', alpha=0.6, linestyle='-', lw=2, label='Global Optimum')
        
    ax2.plot(env.env_space, best_raster_dose, color='red', alpha=0.6, linestyle='-', lw=2, label=best_raster_label)
    ax2.plot(env.env_space, opt_avg_dose, color='blue', alpha=0.8, linestyle='-', lw=2, label=f'Optimiser Dose (MSE: {global_best_mse.item():.3f})')
    ax2.plot(env.env_space, intended_dose, color='black', linestyle='--', lw=2, label="Intended Dose")

    ax2.set_title(f"{'Average ' if starting_phase is None else ''}Dose Distribution")
    ax2.set_xlabel('Position along CTV (cm)')
    ax2.set_ylabel('Relative Dose')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    #Save Figure
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Graph saved to {filepath}")
    
    plt.close(fig)
    
    time.sleep(10) # Avoid thermal throttle
    
if __name__ == "__main__":
    
    # Basic clinical archetypes
    scenarios = [
        {"name": "Mobile_Small_Tumor", "ctv": 1.0, "amp": 2.5},
        {"name": "Static_Large_Tumor", "ctv": 5.0, "amp": 0.5},
        {"name": "Standard_Baseline",  "ctv": 2.0, "amp": 1.0}
    ]
    
    # ---------------------------------------------------------
    print("\n\n=== EXPERIMENT 1: Period vs Repaints ===")
    fixed_spots = 10
    fixed_phase = None
    periods = [2, 3, 5, 7]
    repaints = [0, 1, 4, 9, 19]
    
    for scenario in scenarios:
        save_dir = f"./benchmarks_v2/Exp1_Period_vs_Repaints/{scenario['name']}"
        os.makedirs(save_dir, exist_ok=True)
        for per in periods:
            for rep in repaints:
                optimiser_analysis(
                    ctv_length=scenario["ctv"], amp=scenario["amp"], 
                    period=per, starting_phase=fixed_phase,
                    n_spots=fixed_spots, repaints=rep,
                    save_directory=save_dir
                )

    # ---------------------------------------------------------
    print("\n\n=== EXPERIMENT 2: Repainting===")
    fixed_spots = 10
    repaints = [0, 4]
    periods = [2, 3, 5]
    phases = [None, 0, np.pi/4]
    
    for scenario in scenarios:
        save_dir = f"./benchmarks/Exp2_Starting_Phase_Complexity/{scenario['name']}"
        os.makedirs(save_dir, exist_ok=True)
        for per in periods:
            for phase in phases:
                for repaint in repaints:
                    optimiser_analysis(
                    ctv_length=scenario["ctv"], amp=scenario["amp"], 
                    period=per, starting_phase=phase,
                    n_spots=fixed_spots, repaints=repaint,
                    save_directory=save_dir
                )
                    
    print("\nAll targeted experiments complete! Your data is neatly organized in /benchmarks.")
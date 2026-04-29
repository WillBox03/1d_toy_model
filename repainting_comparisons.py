import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from ITV_engine import ITV_env_2D 
from optimiser import SequenceOptimiser

def run_time_calibrated_sa(opt, target_time_sec, base_pop, eval_weightings):
    """
    Runs a short calibration sprint to measure evals/sec, then calculates and 
    runs the exact number of iterations required to hit the target_time_sec.
    """
    sprint_iters = 200 # Keep this small so calibration is fast
    
    # --- 1. Calibration Sprint ---
    start = time.perf_counter()
    opt.simulated_annealing(iterations=sprint_iters, pop_size=base_pop, 
                            weighting=True, seed_sequence=None, 
                            eval_weightings=eval_weightings)
    sprint_time = time.perf_counter() - start
    
    # --- 2. Calculate Iterations for Target Time ---
    evals_per_sec = (sprint_iters * base_pop) / sprint_time
    total_target_evals = evals_per_sec * target_time_sec
    target_iters = int(total_target_evals / base_pop)
    
    print(f"   [Calibration] Speed: {evals_per_sec:,.0f} evals/sec")
    print(f"   [Calibration] Running {target_iters:,} iterations to hit {target_time_sec}s limit...")
    
    # --- 3. Production Run ---
    run_start = time.perf_counter()
    best_seq, best_cost, _ = opt.simulated_annealing(iterations=target_iters, pop_size=base_pop, 
                                                     weighting=True, seed_sequence=None, 
                                                     eval_weightings=eval_weightings)
    actual_time = time.perf_counter() - run_start
    
    print(f"   [Result] Search finished in {actual_time:.2f} seconds.")
    return best_seq

def evaluate_all_phases(env, sequence, n_phases, period, region, motion):
    """
    Evaluates a delivery sequence across ALL possible starting phases to find 
    the exact Average and the exact Worst Phase (maximum error).
    """
    rmse_list = []
    hi_list = []
    
    for p in range(n_phases):
        dose, _ = env.sim(period=period, sequence=sequence, starting_phase=p, 
                          weighting=False, region=region, motion=motion)
        
        metrics = env.calc_error_metrics(dose, weighting=True, region=region, motion=motion)
        
        # Convert NWMSE to Relative RMSE (%)
        rmse_pct = np.sqrt(metrics.get('MSE (Average)', 0)) * 100.0
        hi = metrics.get('HI (Average)', 1.0)
        
        rmse_list.append(rmse_pct)
        hi_list.append(hi)

    return {
        'RMSE_mean': np.mean(rmse_list),
        'RMSE_std': np.std(rmse_list),
        'Worst_Phase': np.max(rmse_list), # The true Worst Phase Error
        'HI_mean': np.mean(hi_list),
        'HI_std': np.std(hi_list)
    }

def plot_repainting_bars(repaints, raster_data, sa_data, title):
    """
    Plots a 1x3 grid of grouped bar charts showing how metrics change as repaints increase.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.05)

    x = np.arange(len(repaints))
    width = 0.35  # Width of the bars

    # Define subplot configs: (Mean Key, Std Key, Y-Axis Label)
    # Note: 'Worst Phase' is a single maximum value, so it doesn't have a standard deviation.
    metrics_config = [
        ('RMSE_mean', 'RMSE_std', 'Average RMSE (%)'),
        ('Worst_Phase', None, 'Worst Phase RMSE (%)'),
        ('HI_mean', 'HI_std', 'Average HI')
    ]

    for i, (mean_key, std_key, ylabel) in enumerate(metrics_config):
        ax = axes[i]
        
        r_means = raster_data[mean_key]
        s_means = sa_data[mean_key]
        
        r_stds = raster_data[std_key] if std_key else [0]*len(repaints)
        s_stds = sa_data[std_key] if std_key else [0]*len(repaints)
        
        # Draw Raster Bars
        ax.bar(x - width/2, r_means, width, yerr=r_stds if std_key else None, 
               label='Raster Scan', color='#ff6b6b', edgecolor='black', 
               capsize=5, error_kw={'elinewidth': 2})
               
        # Draw Optimised SA Bars
        ax.bar(x + width/2, s_means, width, yerr=s_stds if std_key else None, 
               label='Optimised (SA)', color='#4ecdc4', edgecolor='black', 
               capsize=5, error_kw={'elinewidth': 2})

        ax.set_xlabel('Number of Repaints', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(repaints, fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        
        # Only show legend on the first subplot to save space
        if i == 0:
            ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("repainting_summary_barchart.png", dpi=300, bbox_inches='tight')
    print("-> Saved summary chart: repainting_summary_barchart.png")
    plt.show()
    
def plot_and_save_dose_distributions(env, raster_seq, sa_seq, n_phases, period, region, motion, repaints):
    """
    Extracts the Worst and Random phases for both Raster and SA sequences,
    reshapes them into 2D grids, plots them alongside the Nominal 4D composite, 
    and saves the high-resolution figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. Get 4D Composite and reshape it to 2D
    static_dose = env.get_4d_composite(motion=motion)
    static_dose_2d = static_dose.reshape(env.n_voxels_y, env.n_voxels_x)
    
    # 2. Set color ceiling and physical extents (cm) for axes
    vmax = np.max(static_dose_2d) * 1.2 
    extent = [-env.total_env_half_x, env.total_env_half_x, -env.total_env_half_y, env.total_env_half_y]
    
    raster_doses = []
    sa_doses = []
    raster_mses = []
    sa_mses = []
    
    # 3. Simulate all phases to find the true worst case
    for p in range(n_phases):
        r_dose, _ = env.sim(period=period, sequence=raster_seq, starting_phase=p, weighting=False, region=region, motion=motion)
        s_dose, _ = env.sim(period=period, sequence=sa_seq, starting_phase=p, weighting=False, region=region, motion=motion)
        
        raster_doses.append(r_dose)
        sa_doses.append(s_dose)
        
        raster_mses.append(np.mean((r_dose - static_dose)**2))
        sa_mses.append(np.mean((s_dose - static_dose)**2))

    # 4. Identify indices
    r_worst_idx = np.argmax(raster_mses)
    s_worst_idx = np.argmax(sa_mses)
    rand_idx = np.random.randint(0, n_phases)

    # 5. Setup 1x5 Plot
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    fig.suptitle(f"Interplay Mitigation: {repaints} Repaints", fontsize=16, fontweight='bold', y=1.05)
    
    # Bundle the data and reshape directly in the list
    plots_data = [
        ("Target 4D Composite", static_dose_2d),
        (f"Raster (Random Phase {rand_idx})", raster_doses[rand_idx].reshape(env.n_voxels_y, env.n_voxels_x)),
        (f"Optimised SA (Random Phase {rand_idx})", sa_doses[rand_idx].reshape(env.n_voxels_y, env.n_voxels_x)),
        (f"Raster (Worst Phase {r_worst_idx})", raster_doses[r_worst_idx].reshape(env.n_voxels_y, env.n_voxels_x)),
        (f"Optimised SA (Worst Phase {s_worst_idx})", sa_doses[s_worst_idx].reshape(env.n_voxels_y, env.n_voxels_x))
    ]
    
    # 6. Plot each distribution
    for ax, (title, dose_2d) in zip(axes, plots_data):
        im = ax.imshow(dose_2d, cmap='jet', vmin=0, vmax=vmax, extent=extent, origin='lower')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("X (cm)", fontsize=10)
        
        # Only put the Y-axis label and ticks on the first plot to keep it clean
        if ax == axes[0]:
            ax.set_ylabel("Y (cm)", fontsize=10)
        else:
            ax.set_yticks([]) 
            
    # Add a single shared colorbar to the right
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Dose (Arbitrary Units)')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # 7. Save the figure
    filename = f"dose_distributions_{repaints}_repaints.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   -> Saved dose map: {filename}")
    plt.close()
    
def save_improvement_summary(repaints_list, raster_history, sa_history):
    """
    Calculates the percentage improvement of SA over Raster for each metric 
    and saves the result to a CSV file.
    """
    summary_data = []

    for i, r in enumerate(repaints_list):
        # Calculate Percentage Improvements
        # (Raster - SA) / Raster * 100
        # Positive result = SA is better (lower error)
        
        rmse_impr = ((raster_history['RMSE_mean'][i] - sa_history['RMSE_mean'][i]) / 
                     raster_history['RMSE_mean'][i]) * 100
        
        worst_impr = ((raster_history['Worst_Phase'][i] - sa_history['Worst_Phase'][i]) / 
                      raster_history['Worst_Phase'][i]) * 100
        
        # For HI, the ideal value is 0.0, so we treat it the same way
        hi_impr = ((raster_history['HI_mean'][i] - sa_history['HI_mean'][i]) / 
                   raster_history['HI_mean'][i]) * 100

        summary_data.append({
            'Repaints': r,
            'RMSE Improvement (%)': round(rmse_impr, 2),
            'Worst Phase Improvement (%)': round(worst_impr, 2),
            'HI Improvement (%)': round(hi_impr, 2),
            'Absolute Raster RMSE': round(raster_history['RMSE_mean'][i], 2),
            'Absolute SA RMSE': round(sa_history['RMSE_mean'][i], 2)
        })

    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    filename = "optimizer_performance_summary.csv"
    df.to_csv(filename, index=False)
    
    print("\n" + "="*30)
    print("STATISTICAL SUMMARY")
    print("="*30)
    print(df.to_string(index=False))
    print(f"\n-> Saved detailed stats to: {filename}")

if __name__ == "__main__":
    # Setup Parameters
    ctv_size = (2.0, 2.0)
    n_spots = (5, 10)
    spot_size = (0.4, 0.4)
    amp = (0, 1)      
    res = 0.05   
    t_steps = (0.05, 0.05)
    period = 5.0
    margin = (0.5, 2)
    region = 'itv'
    motion = 'sin6'
    n_phases = 24
    
    target_search_time = 30.0 # Strict 30s benchmark limit
    base_pop_size = 24
    
    repaints_list = [0, 1, 3, 9, 19]
    
    raster_history = {'RMSE_mean': [], 'RMSE_std': [], 'Worst_Phase': [], 'HI_mean': [], 'HI_std': []}
    sa_history = {'RMSE_mean': [], 'RMSE_std': [], 'Worst_Phase': [], 'HI_mean': [], 'HI_std': []}

    print("Initializing Simulation Environment...")
    env = ITV_env_2D(ctv_size=ctv_size, n_spots=n_spots, spot_size=spot_size, amp=amp, 
                     res=res, t_steps=t_steps, margin=margin)

    for r in repaints_list:
        print("\n" + "="*50)
        print(f"RUNNING EXPERIMENT: {r} REPAINTS")
        print("="*50)
        
        env.set_spot_weights('uniform', repaints=r)
        
        # 1. Raster Setup & Evaluation
        print(f"Simulating Raster Scan ({r} repaints)...")
        raster_seq = env.set_sequence('lr_rast')
        r_stats = evaluate_all_phases(env, raster_seq, n_phases, period, region, motion)
        
        # 2. SA Setup & Calibrated Search
        print(f"Optimizing SA Sequence ({target_search_time}s benchmark)...")
        env.calculate_mask_tensor(period=period, starting_phase=None, region=region, motion=motion)
        opt = SequenceOptimiser(env)
        
        eval_weightings = [1000, 1000 + (10 * r), 2, 0] # Your custom weightings
        
        # Run the benchmarked wrapper we built earlier
        best_sa_seq = run_time_calibrated_sa(opt, target_time_sec=target_search_time, 
                                         base_pop=base_pop_size, 
                                         eval_weightings=eval_weightings)
        
        s_stats = evaluate_all_phases(env, best_sa_seq, n_phases, period, region, motion)
        
        # 3. Store Stats for Bar Chart
        for key in raster_history.keys():
            raster_history[key].append(r_stats[key])
            sa_history[key].append(s_stats[key])
            
        # 4. Generate Visual Plots for specific repaints
        if r in [0, 1, 19]:
            print(f"Generating visual dose maps for {r} repaints...")
            plot_and_save_dose_distributions(env, raster_seq, best_sa_seq, 
                                             n_phases, period, region, motion, repaints=r)

    # 5. Generate and Save Final Bar Chart
    print("\nGenerating Final Repainting Impact Graph...")
    plot_repainting_bars(repaints_list, raster_history, sa_history, title="Impact of Repainting on Interplay Mitigation")
    
    save_improvement_summary(repaints_list, raster_history, sa_history)
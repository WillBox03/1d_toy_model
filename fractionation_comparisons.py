import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from ITV_engine import ITV_env_2D 
from optimiser import SequenceOptimiser

# ==========================================
# 1. CORE SIMULATION FUNCTIONS
# ==========================================

def simulate_fractionation_course(env, sequence, n_fractions, n_trials, n_phases, period, region, motion):
    """
    Simulates a clinical fractionation course over 'n_trials' independent runs.
    Calculates metrics on the accumulated average dose.
    """
    trial_metrics = {
        'MSE (Average)': [],
        'HI (Average)': [],
        'Worst Voxel Error (CTV Average)': [] 
    }
    
    # Track the accumulated dose of the final trial for plotting
    final_accumulated_dose = None 
    
    for trial in range(n_trials):
        accumulated_dose = np.zeros(env.n_env_voxels, dtype=np.float64)
        
        for fraction in range(n_fractions):
            random_p = np.random.randint(0, n_phases)
            f_dose, _ = env.sim(period=period, sequence=sequence, starting_phase=random_p, 
                                weighting=False, region=region, motion=motion)
            accumulated_dose += f_dose
            
        average_course_dose = accumulated_dose / n_fractions
        metrics = env.calc_error_metrics(average_course_dose, weighting=True, region=region, motion=motion)
        
        rmse_pct = np.sqrt(metrics.get('MSE (Average)', 0)) * 100.0
        worst_pct = metrics.get('Worst Voxel Error (CTV Average)', 0) * 100.0
        
        trial_metrics['MSE (Average)'].append(rmse_pct)
        trial_metrics['Worst Voxel Error (CTV Average)'].append(worst_pct)
        trial_metrics['HI (Average)'].append(metrics.get('HI (Average)', 1.0))
        
        if trial == n_trials - 1:
            final_accumulated_dose = average_course_dose

    summary_stats = {}
    for key in trial_metrics.keys():
        summary_stats[key] = {
            'mean': np.mean(trial_metrics[key]),
            'std': np.std(trial_metrics[key])
        }
        
    return summary_stats, final_accumulated_dose

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
# ==========================================
# 2. PLOTTING AND SAVING FUNCTIONS
# ==========================================

def plot_fractionation_bars(repaints_list, raster_data, sa_data, regime_name, n_fractions, env, motion, region):
    """
    Plots the bar charts with the updated repaint list [0, 3, 9, 19].
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{regime_name} ({n_fractions} Fractions) Comparison", fontsize=18, fontweight='bold', y=1.05)

    x = np.arange(len(repaints_list))
    width = 0.35

    metrics_config = [
        ('MSE (Average)', 'Relative RMSE (%)'),
        ('Worst Voxel Error (CTV Average)', 'Worst CTV Error (%)'),
        ('HI (Average)', 'Average HI')
    ]

    for i, (key, ylabel) in enumerate(metrics_config):
        ax = axes[i]
        
        r_means = [raster_data[r][key]['mean'] for r in repaints_list]
        r_stds = [raster_data[r][key]['std'] for r in repaints_list]
        s_means = [sa_data[r][key]['mean'] for r in repaints_list]
        s_stds = [sa_data[r][key]['std'] for r in repaints_list]
        
        ax.bar(x - width/2, r_means, width, yerr=r_stds, label='Raster', color='#ff6b6b', edgecolor='black', capsize=5)
        ax.bar(x + width/2, s_means, width, yerr=s_stds, label='SA', color='#4ecdc4', edgecolor='black', capsize=5)

        ax.set_xticks(x)
        ax.set_xticklabels(repaints_list)
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_xlabel('Number of Repaints', fontweight='bold')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"barchart_{regime_name.lower()}.png", dpi=300)
    plt.close()

def plot_dynamic_dose_maps(env, raster_doses, sa_doses, repaints_list, regime_name, motion):
    """
    Plots a 2xN grid where each column is normalized to its own 
    specific 4D Composite (Prescription Dose).
    """
    n_cols = len(repaints_list)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    fig.suptitle(f"{regime_name} Dose Distributions (Normalized to Local $D_{{rx}}$)", 
                 fontsize=18, fontweight='bold', y=0.98)
    
    extent = [-env.total_env_half_x, env.total_env_half_x, -env.total_env_half_y, env.total_env_half_y]

    for col, r in enumerate(repaints_list):
        # --- LOCAL NORMALIZATION ---
        # Update the env temporarily to get the correct 4D composite for THIS repaint level
        env.set_spot_weights('uniform', repaints=r)
        local_4d_composite = env.get_4d_composite(motion=motion)
        
        # Prescription Dose (Max of the motion-blurred target)
        drx = np.max(local_4d_composite)
        local_vmax = drx * 1.2 # Give 20% headroom for hot spots
        
        # Row 0: Raster
        ax_r = axes[0, col]
        dose_r = raster_doses[r].reshape(env.n_voxels_y, env.n_voxels_x)
        im = ax_r.imshow(dose_r, cmap='jet', vmin=0, vmax=local_vmax, extent=extent, origin='lower')
        ax_r.set_title(f"Raster ({r} Repaints)\n$D_{{rx}}$={drx:.1f}", fontsize=11, fontweight='bold')
        
        # Row 1: SA
        ax_s = axes[1, col]
        dose_s = sa_doses[r].reshape(env.n_voxels_y, env.n_voxels_x)
        ax_s.imshow(dose_s, cmap='jet', vmin=0, vmax=local_vmax, extent=extent, origin='lower')
        ax_s.set_title(f"SA ({r} Repaints)\n$D_{{rx}}$={drx:.1f}", fontsize=11, fontweight='bold')

        # Clean up axes
        for ax in [ax_r, ax_s]:
            ax.set_xlabel("X (cm)")
            if col == 0: ax.set_ylabel("Y (cm)")
            else: ax.set_yticks([])

    # Colorbar here represents Relative Dose (0% to 120% of Local Prescription)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Relative Dose Magnitude', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    filename = f"dosemap_{regime_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"-> Saved Normalized Dosemap: {filename}")
    plt.close()

def save_regime_csv(repaints_list, raster_data, sa_data, regime_name):
    """Saves percentage improvements to a CSV."""
    data = []
    keys = ['MSE (Average)', 'Worst Voxel Error (CTV Average)', 'HI (Average)']
    for r in repaints_list:
        row = {'Regime': regime_name, 'Repaints': r}
        for k in keys:
            r_val = raster_data[r][k]['mean']
            s_val = sa_data[r][k]['mean']
            impr = ((r_val - s_val) / r_val) * 100 if r_val > 0 else 0
            row[f'{k} Raster'] = round(r_val, 4)
            row[f'{k} SA'] = round(s_val, 4)
            row[f'{k} Impr (%)'] = round(impr, 2)
        data.append(row)
        
    df = pd.DataFrame(data)
    filename = f"summary_{regime_name.lower().replace(' ', '_')}.csv"
    df.to_csv(filename, index=False)
    print(f"-> Saved CSV: {filename}")

# ==========================================
# 3. MAIN EXECUTION LOOP
# ==========================================

if __name__ == "__main__":
    # Standard Setup
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
    
    target_search_time = 30.0 # Strict 30s limit
    n_trials = 50
    
    # ADDED 3 REPAINTS
    repaints_list = [0, 3, 9, 19]
    
    regimes = {
        'Hypofractionation': 5,
        'Hyperfractionation': 30
    }

    print("Initializing Simulation Environment...")
    env = ITV_env_2D(ctv_size=ctv_size, n_spots=n_spots, spot_size=spot_size, amp=amp, 
                     res=res, t_steps=t_steps, margin=margin)

    for regime_name, n_fracs in regimes.items():
        print("\n" + "="*60)
        print(f"STARTING {regime_name.upper()} ({n_fracs} Fractions)")
        print("="*60)
        
        raster_stats = {}
        sa_stats = {}
        raster_doses = {}
        sa_doses = {}

        for r in repaints_list:
            print(f"\n--- Testing {r} Repaints ---")
            env.set_spot_weights('uniform', repaints=r)
            
            print(f"   Simulating Raster...")
            raster_seq = env.set_sequence('lr_rast')
            r_stat, r_dose = simulate_fractionation_course(env, raster_seq, n_fractions=n_fracs, n_trials=n_trials, 
                                                           n_phases=n_phases, period=period, region=region, motion=motion)
            raster_stats[r] = r_stat
            raster_doses[r] = r_dose
            
            print(f"   Optimising SA ({target_search_time}s limit)...")
            env.calculate_mask_tensor(period=period, starting_phase=None, region=region, motion=motion)
            opt = SequenceOptimiser(env)
            eval_weightings = [1000, 1000 + (10 * r), 2, 0] 
            
            best_sa_seq = run_time_calibrated_sa(opt, target_time_sec=target_search_time, base_pop=24, eval_weightings=eval_weightings)
            
            print(f"   Simulating SA Fractionation...")
            s_stat, s_dose = simulate_fractionation_course(env, best_sa_seq, n_fractions=n_fracs, n_trials=n_trials, 
                                                           n_phases=n_phases, period=period, region=region, motion=motion)
            sa_stats[r] = s_stat
            sa_doses[r] = s_dose

        # Generate outputs for this regime
        print(f"\nGenerating Graphs for {regime_name}...")
        
        # Note: Added env, motion, region parameters for 4D baseline calculation
        plot_fractionation_bars(repaints_list, raster_stats, sa_stats, regime_name, n_fracs, env, motion, region)
        plot_dynamic_dose_maps(env, raster_doses, sa_doses, repaints_list, regime_name, motion)
        save_regime_csv(repaints_list, raster_stats, sa_stats, regime_name)
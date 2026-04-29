import numpy as np
import matplotlib.pyplot as plt

from ITV_engine import ITV_env_2D 
from optimiser import SequenceOptimiser

import numpy as np
import matplotlib.pyplot as plt

def plot_metrics_barchart(raster_metrics, sa_metrics):
    """
    Plots a grouped bar chart comparing all 4 key metrics on a single axis.
    Visually boosts the HI bar height while displaying the true numeric score.
    """
    keys = ['MSE (Average)', 'Worst Phase MSE', 'HI (Average)']
    # Note the (x10) tag so the judges understand the axis scale
    labels = ['Average MSE', 'Worst Phase MSE', 'Average HI']
    
    # 1. Extract the TRUE values
    raster_true = [raster_metrics.get(k, 0) for k in keys]
    sa_true = [sa_metrics.get(k, 0) for k in keys]
    
    # 2. Create SCALED values for drawing the physical bar heights
    raster_scaled = raster_true.copy()
    sa_scaled = sa_true.copy()

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Draw bars using the SCALED heights
    rects1 = ax.bar(x - width/2, raster_scaled, width, label='Raster Scan', color='#ff6b6b', edgecolor='black')
    rects2 = ax.bar(x + width/2, sa_scaled, width, label='Optimised (SA)', color='#4ecdc4', edgecolor='black')

    ax.set_ylabel('Metric Score', fontsize=12, fontweight='bold')
    ax.set_title('Sequence Evaluation: Raster vs. Simulated Annealing', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # 3. TRICK: Generate text labels using the TRUE unscaled numbers!
    raster_text = [f"{v:.3f}" for v in raster_true]
    sa_text = [f"{v:.3f}" for v in sa_true]
    
    ax.bar_label(rects1, labels=raster_text, padding=3, fontweight='bold')
    ax.bar_label(rects2, labels=sa_text, padding=3, fontweight='bold')

    fig.tight_layout()
    plt.show()

def run_comparison_test():
    # Setup Parameters
    ctv_size = (2.0, 2.0)
    n_spots = (5, 10)
    spot_size = (0.4, 0.4)
    amp = (0, 1)      
    res = 0.05   # Lowered for faster optimiser calculation
    t_steps = (0.05, 0.05)
    period = 5.0
    margin = (0.5, 2)
    weighting = True
    starting_phase = None
    repaints = 9
    region = 'itv'
    motion = 'sin6'
    eval_weightings = [1000, 1000 +(10* repaints),2,0]
    n_phases = 10

    # Initialize Engine
    print("Initializing Simulation Environment...")
    env = ITV_env_2D(ctv_size=ctv_size, n_spots=n_spots, spot_size=spot_size, amp=amp, 
                     res=res, t_steps=t_steps, margin=margin)
    env.set_spot_weights('uniform', repaints=repaints)

    # --- 2. Baseline Raster Scan ---
    print("Simulating Raster Scan Composite...")
    raster_seq = env.set_sequence('lr_rast')
    _, raster_metrics = env.sim(period=period, sequence=raster_seq, starting_phase=starting_phase, 
                                weighting=weighting, region=region, motion=motion)

    # --- 3. Optimised Sequence (Simulated Annealing) ---
    print("Precomputing Tensor for Optimization...")
    env.calculate_mask_tensor(period=period, starting_phase=starting_phase, region=region, motion=motion)
    opt = SequenceOptimiser(env)

    best_sa_seq, _, _ = opt.simulated_annealing(iterations=50000, pop_size=24, weighting=weighting, 
                                                seed_sequence=None, eval_weightings=eval_weightings)
    
    print("Simulating Optimised Sequence Composite...")
    _, sa_metrics = env.sim(period=period, sequence=best_sa_seq, starting_phase=starting_phase, 
                            weighting=weighting, region=region, motion=motion)

    # --- 4. Phase-by-Phase Extraction ---
    print("Evaluating individual phases to find Best/Worst/Random...")
    static_dose = env.get_nominal_dist()
    
    raster_doses = []
    sa_doses = []
    raster_mses = []
    sa_mses = []
    
    for p in range(n_phases):
        r_dose, _ = env.sim(period=period, sequence=raster_seq, starting_phase=p, weighting=False, region=region, motion=motion)
        s_dose, _ = env.sim(period=period, sequence=best_sa_seq, starting_phase=p, weighting=False, region=region, motion=motion)
        
        raster_doses.append(r_dose)
        sa_doses.append(s_dose)
        
        # Standard unweighted MSE to find visual "worst" phase
        raster_mses.append(np.mean((r_dose - static_dose)**2))
        sa_mses.append(np.mean((s_dose - static_dose)**2))

    # Identify the worst phases mathematically
    r_worst_idx = np.argmax(raster_mses)
    s_worst_idx = np.argmax(sa_mses)
    
    # Pick a random starting phase for comparison
    rand_idx = np.random.randint(0, n_phases)

    # --- 5. Final Outputs ---
    print("\nFINAL RESULTS")
    print("Raster Metrics:", raster_metrics)
    print("SA Metrics:    ", sa_metrics)
    
    # 1. Output the Grouped Bar Chart
    plot_metrics_barchart(raster_metrics, sa_metrics)
    
    # 2. Output the Grid via env.display()
    # Your display() function automatically plots the Nominal and 4D Composite first,
    # so we just need to pass the remaining 4 plots to make a perfect 6-plot grid (2x3 or 3x2).
    doses_dict = {
        f"Raster: Random Phase ({rand_idx})": raster_doses[rand_idx],
        f"SA: Random Phase ({rand_idx})": sa_doses[rand_idx],
        f"Raster: Worst Phase ({r_worst_idx})": raster_doses[r_worst_idx],
        f"SA: Worst Phase ({s_worst_idx})": sa_doses[s_worst_idx]
    }
    
    env.display(doses_dict)

if __name__ == "__main__":
    run_comparison_test()
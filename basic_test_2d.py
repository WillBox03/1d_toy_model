import numpy as np

from ITV_engine import ITV_env_2D 
from optimiser import SequenceOptimiser

def run_comparison_test():
    # Setup Parameters
    ctv_size = (2.0, 2.0)
    n_spots = (5, 10)
    spot_size = (0.4, 0.4)
    amp = (0.0, 1)      
    res = 0.05    # Lowered for faster optimiser calculation
    t_steps = (0.05, 0.05)
    period = 3.0
    margin = (0.5, 1.5)
    weighting = True
    starting_phase = None
    repaints = 9
    region = 'itv'
    motion = 'sin6'
    eval_weightings = [2,2.5*repaints,2, 1]

    # Initialize Engine
    env = ITV_env_2D(ctv_size=ctv_size, n_spots=n_spots, spot_size = spot_size, amp=amp, 
                      res=res, t_steps=t_steps, margin=margin)
    
    env.set_spot_weights('uniform', repaints= repaints)
    # Baseline plots
    # Baseline Raster
    raster_seq = env.set_sequence('lr_rast')
    raster_dose, raster_metrics = env.sim(period=period, sequence=raster_seq, starting_phase=starting_phase, weighting=weighting, region = region, motion=motion)
    raster_mse = raster_metrics['MSE (Average)']
    # Random Sequence
    rand_seq = env.set_sequence('rand')
    rand_dose, rand_metrics = env.sim(period=period, sequence=rand_seq, starting_phase=starting_phase, weighting=weighting, region = region, motion=motion)
    rand_mse = rand_metrics['MSE (Average)']
    # Maximum Distance/Time Sequence
    max_seq = env.set_sequence('max_dist')
    max_dose, max_metrics = env.sim(period=period, sequence=max_seq, starting_phase=starting_phase, weighting=weighting, region = region, motion=motion)
    max_mse = max_metrics['MSE (Average)']
    # Tensor Calculation
    print("Precomputing Tensor")
    env.calculate_mask_tensor(period=period, starting_phase=starting_phase, region = region, motion=motion)
    opt = SequenceOptimiser(env)

    # Optimisation Methods
    # Monte Carlo
    mc_seq, mc_score, _ = opt.monte_carlo(n_samples=100000, weighting=weighting, eval_weightings = eval_weightings)
    mc_dose, mc_metrics = env.sim(period=period, sequence=mc_seq, starting_phase=starting_phase, weighting=weighting, region = region, motion=motion)
    mc_mse = mc_metrics['MSE (Average)']
    # Simulated Annealing
    best_sa_seq, sa_score, _ = opt.simulated_annealing(iterations=10000, pop_size=24, weighting=weighting, seed_sequence = None, eval_weightings = eval_weightings)
    sa_dose, sa_metrics = env.sim(period=period, sequence=best_sa_seq, starting_phase=starting_phase, weighting=weighting, region = region, motion=motion)
    sa_mse = sa_metrics['MSE (Average)']
    # Results
    print("FINAL RESULTS (MSE)")
    print(f"Raster:             {raster_mse:.6f}")
    print(f"Random:             {rand_mse:.6f}")
    print(f"Max Distance/Time:  {max_mse:.6f}")
    print(f"Monte Carlo:        {mc_mse:.6f}")
    print(f"Simulated Annealing:{sa_mse:.6f}")

    # Organise display into grid
    env.display({
        f"Raster\n(MSE: {raster_mse:.4f})": raster_dose,
        f"Random\n(MSE: {rand_mse:.4f})": rand_dose,
        f"Max Dist\n(MSE: {max_mse:.4f})": max_dose,
        f"Monte Carlo\n(MSE: {mc_mse:.4f})": mc_dose,
        f"Sim Annealing\n(MSE: {sa_mse:.4f})": sa_dose
    })
    print(raster_metrics)
    print(sa_metrics)
    
    #env.display_dvh(dose, regions = ['itv', 'ctv'], show_dvh = False)

if __name__ == "__main__":
    run_comparison_test()
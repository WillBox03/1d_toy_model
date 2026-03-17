from ITV_engine import ITV_env_2D 
from optimiser import SequenceOptimiser

def run_comparison_test():
    # Setup Parameters
    ctv_size = (2.0, 2.0)
    n_spots = (10, 15)
    amp = (0, 0.5)      
    res = 0.05       # Lowered for faster optimiser calculation
    t_steps = (0.1, 0.1)
    period = 7.0
    margin = (1.0, 0.5)
    weighting = False
    starting_phase = None
    region = 'itv'

    # Initialize Engine
    env = ITV_env_2D(ctv_size=ctv_size, n_spots=n_spots, amp=amp, 
                      res=res, t_steps=t_steps, margin=margin)
    
    env.set_spot_weights('uniform', repaints= 2)
    
    # Baseline plots
    # Baseline Raster
    raster_seq = env.set_sequence('lr_rast')
    raster_dose, raster_err = env.sim(period=period, sequence=raster_seq, starting_phase=starting_phase, weighting=weighting, region = region)

    # Random Sequence
    rand_seq = env.set_sequence('rand')
    rand_dose, rand_err = env.sim(period=period, sequence=rand_seq, starting_phase=starting_phase, weighting=weighting, region = region)
    
    # Maximum Distance/Time Sequence
    max_seq = env.set_sequence('max_dist')
    max_dose, max_err = env.sim(period=period, sequence=max_seq, starting_phase=starting_phase, weighting=weighting, region = region)

    # Tensor Calculation
    print("Precomputing Tensor")
    env.calculate_mask_tensor(period=period, starting_phase=starting_phase, region = region)
    opt = SequenceOptimiser(env)

    # Optimisation Methods
    # Monte Carlo
    mc_seq, mc_err, _ = opt.monte_carlo(n_samples=100000, weighting=weighting)
    mc_dose, _ = env.sim(period=period, sequence=mc_seq, starting_phase=starting_phase, weighting=weighting, region = region)

    # Simulated Annealing
    best_sa_seq, sa_err, _ = opt.simulated_annealing(iterations=10000, pop_size=30, weighting=weighting)
    sa_dose, _ = env.sim(period=period, sequence=best_sa_seq, starting_phase=starting_phase, weighting=weighting, region = region)

    # Results
    print("FINAL RESULTS (MSE)")
    print(f"Raster:             {raster_err:.6f}")
    print(f"Random:             {rand_err:.6f}")
    print(f"Max Distance/Time:  {max_err:.6f}")
    print(f"Monte Carlo:        {mc_err:.6f}")
    print(f"Simulated Annealing:{sa_err:.6f}")
    
    # Organise display into grid
    env.display({
        f"Raster\n(MSE: {raster_err:.4f})": raster_dose,
        f"Random\n(MSE: {rand_err:.4f})": rand_dose,
        f"Max Dist\n(MSE: {max_err:.4f})": max_dose,
        f"Monte Carlo\n(MSE: {mc_err:.4f})": mc_dose,
        f"Sim Annealing\n(MSE: {sa_err:.4f})": sa_dose
    })

if __name__ == "__main__":
    run_comparison_test()
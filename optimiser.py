import numpy as np
from math import factorial
from numba import njit, prange
import time

from ITV_engine import _evaluation_core_single_thread

@njit(cache = True)
def _process_sequence(tensor, intended, seq, seq_len, times, weighting, t_step_ticks, mode_pen):
    
    # Find the times of the sequence
    times[0] = 0
    for s in range(1, seq_len):
        
        dist = np.int16(abs(seq[s] - seq[s-1]))
        
        ticks = (dist * t_step_ticks) + ((dist == 0) * mode_pen)
        
        times[s] = times[s-1] + ticks
    # Evaluate sequence    
    return _evaluation_core_single_thread(tensor, intended, seq, times, weighting)
    

@njit(parallel=True, cache=True)
def _sa_core(tensor, intended, start_seqs, iterations, temp, c_fact, weighting, n_bins, t_step_ticks, mode_penalty):
    
    pop_size = start_seqs.shape[0]
    seq_len = start_seqs.shape[1]
    
    final_best_seqs = np.zeros_like(start_seqs)
    final_best_mses = np.zeros(pop_size)
    
    # Pre-allocate all debug matrices
    all_mse_walkers = np.zeros((pop_size, iterations), dtype=np.float64)
    all_acceptance_bins = np.zeros((pop_size, n_bins), dtype=np.float64)
    total_accepted = np.zeros(pop_size, dtype=np.int32)
    
    bin_size = iterations / n_bins
    
    # Split into parallel threads
    for p in prange(pop_size):
        # Initialise sequencing variables
        current_seq = start_seqs[p].copy()
        test_seq = start_seqs[p].copy()
        best_seq = start_seqs[p].copy()
        times = np.zeros(seq_len, dtype=np.int16)
        
        # Evaluate initial sequence    
        current_mse = _process_sequence(tensor, intended, current_seq, seq_len, times, weighting, t_step_ticks, mode_penalty)
        best_mse = current_mse
        
        T = temp 
        
        accepted_count = 0 #Debugging param for acceptance rate
        
        # Main SA chain
        for i in range(iterations):
            # Perform random swap
            idx1 = np.random.randint(0, seq_len)
            idx2 = np.random.randint(0, seq_len)
            while idx1 == idx2 or current_seq[idx1] == current_seq[idx2]:
                idx2 = np.random.randint(0, seq_len)
                
            test_seq[idx1], test_seq[idx2] = test_seq[idx2], test_seq[idx1]
            
            test_mse = _process_sequence(tensor, intended, test_seq, seq_len, times, weighting, t_step_ticks, mode_penalty)
            diff = test_mse - current_mse
            
            accepted_iter = 0 # Acceptance param
            
            # Acceptance condition
            if diff < 0 or np.random.rand() < np.exp(-(diff / current_mse) / T):
                # Accept
                for s in range(seq_len):
                    current_seq[s] = test_seq[s]
                current_mse = test_mse
                accepted_iter = 1
                accepted_count += 1
                # Check if new sequence is the best
                if current_mse < best_mse:
                    best_mse = current_mse
                    for s in range(seq_len):
                        best_seq[s] = current_seq[s]
            else:
                # Reject
                test_seq[idx1], test_seq[idx2] = test_seq[idx2], test_seq[idx1]
                
            T *= c_fact # Cool temp    
            
            # Record debug info for the thread
            all_mse_walkers[p, i] = current_mse
            if accepted_iter > 0:
                bin_idx = int(min(i // bin_size, n_bins - 1))
                all_acceptance_bins[p, bin_idx] += 1
            
        # Save best sequence in thread
        final_best_mses[p] = best_mse
        for s in range(seq_len):
            final_best_seqs[p, s] = best_seq[s]
            
        total_accepted[p] = accepted_count

    return final_best_seqs, final_best_mses, all_acceptance_bins, all_mse_walkers, total_accepted

@njit(parallel=True, cache=True)
def _numba_mc_core(tensor, intended, base_seq, n_samples, weighting, t_step_ticks, mode_penalty):
    
    seq_len = len(base_seq)
    # Preallocation
    seqs = np.zeros((n_samples, seq_len), dtype=np.int16)
    evals = np.zeros(n_samples, dtype=np.float64)
    
    # Parallel evaluation of random sequences
    for i in prange(n_samples):
        seq = np.random.permutation(base_seq)

        times = np.zeros(seq_len, dtype=np.int16) # Preallocate times
        # Evaluate sequences
        evals[i] = _process_sequence(tensor, intended, seq, seq_len, times, weighting, t_step_ticks, mode_penalty)
        for s in range(seq_len):
            seqs[i, s] = seq[s]
            
    return seqs, evals

@njit(parallel = True, cache = True)
def _local_search_core(tensor, intended, pop_seqs, weighting, t_step_ticks, mode_penalty):
    
    pop_size = pop_seqs.shape[0]
    seq_len = pop_seqs.shape[1]
    # Preallocation
    best_neighbour_seqs = np.zeros_like(pop_seqs)
    best_neighbour_mses = np.full(pop_size, np.inf)
    
    # Split into parallel threads
    for p in prange(pop_size):
        # Initialise sequencing variables
        test_seq = pop_seqs[p].copy()
        best_local_seq = pop_seqs[p].copy()
        best_local_mse = np.inf
        times = np.zeros(seq_len, dtype = np.int16)
        
        # Loop through all swaps
        for i in range(seq_len - 1):
            for j in range(i + 1, seq_len):
                if test_seq[i] == test_seq[j]:
                    continue
                # Swap spots    
                test_seq[i], test_seq[j] = test_seq[j], test_seq[i]
                
                # Evaluate swap    
                mse = _process_sequence(tensor, intended, test_seq, seq_len, times, weighting, t_step_ticks, mode_penalty)
                
                # Update if new sequence is best
                if mse < best_local_mse:
                    best_local_mse = mse
                    for s in range(seq_len):
                        best_local_seq[s] = test_seq[s]
                        
                # Swap back before generating new swap    
                test_seq[i], test_seq[j] = test_seq[j], test_seq[i]
        
        # Finalise best local sequences    
        best_neighbour_mses[p] = best_local_mse
        for s in range(seq_len):
            best_neighbour_seqs[p, s] = best_local_seq[s]
            
    return best_neighbour_seqs, best_neighbour_mses

@njit(cache=True)
def _hybrid_generation_loop(tensor, intended, base_seq, pop_seqs, pop_evals, generations, n_samples, weighting, t_step_ticks, mode_penalty):
    pop_size = pop_seqs.shape[0]
    seq_len = pop_seqs.shape[1]
    
    best_seqs = pop_seqs.copy()
    best_evals = pop_evals.copy()
    
    # Preallocate pool arrays for combined best and mc sequences
    pool_size = 2 * pop_size
    combined_seqs = np.zeros((pool_size, seq_len), dtype=np.int16)
    combined_evals = np.zeros(pool_size, dtype=np.float64)

    for gen in range(generations):
        # Local search
        neighbor_seqs, neighbor_mses =  _local_search_core(tensor, intended, best_seqs, weighting, t_step_ticks, mode_penalty)
        
        num_stagnant = 0
        stagnant_indices = np.zeros(pop_size, dtype=np.int32)
        
        # Add improvements, flag stagnations
        for i in range(pop_size):
            if neighbor_mses[i] < best_evals[i]:
                # Improvements are added
                best_evals[i] = neighbor_mses[i]
                for s in range(seq_len):
                    best_seqs[i, s] = neighbor_seqs[i, s]
            else:
                # Flag 
                if i != 0: 
                    stagnant_indices[num_stagnant] = i
                    num_stagnant += 1
                    
        # If stagnant sequences, replace
        if num_stagnant > 0:
            # Generate MC samples for replacement
            n_new_samples = int((n_samples // pop_size) * num_stagnant)
            new_seqs, new_evals = _numba_mc_core(tensor, intended, base_seq, n_new_samples, weighting, t_step_ticks, mode_penalty)
            
            # Sort the best ones
            top_new_idx = np.argsort(new_evals)
            
            # Overwrite stagnant sequences
            for k in range(num_stagnant):
                target_idx = stagnant_indices[k]
                best_evals[target_idx] = new_evals[top_new_idx[k]]
                for s in range(seq_len):
                    best_seqs[target_idx, s] = new_seqs[top_new_idx[k], s]
                    
        # New global MC samples to replace best sequences
        global_new_seqs, global_new_evals = _numba_mc_core(tensor, intended, base_seq, n_samples, weighting, t_step_ticks, mode_penalty)
        # Sort the indexes
        global_top_idx = np.argsort(global_new_evals)
        
        # Combine best sequences and mc_sequences
        # Starting with best sequences
        for i in range(pop_size):
            combined_evals[i] = best_evals[i]
            for s in range(seq_len):
                combined_seqs[i, s] = best_seqs[i, s]
        # Then new MC sequences (only add the top 10)        
        for i in range(pop_size):
            best_mc_index = global_top_idx[i]
            combined_evals[pop_size + i] = global_new_evals[best_mc_index]
            for s in range(seq_len):
                combined_seqs[pop_size + i, s] = global_new_seqs[best_mc_index, s]
                
        # Sort and replace weaker sequences with new MC sequences
        top_idx = np.argsort(combined_evals)
        for i in range(pop_size):
            best_evals[i] = combined_evals[top_idx[i]]
            for s in range(seq_len):
                best_seqs[i, s] = combined_seqs[top_idx[i], s]

    # Return the global best
    return best_seqs[0], best_evals[0]

@njit(parallel = True, cache = True)
def _exhaustive_core(tensor, intended, base_seq, weighting, t_step_ticks, mode_penalty):
    
    n = len(base_seq)
    
    # Pre-allocate arrays
    best_seqs = np.zeros((n, n), dtype=np.int16)
    best_mses = np.full(n, np.inf, dtype=np.float64)
    
    # Split the N! permutations across threads by swapping the first element
    for first_idx in prange(n):
        
        seq = base_seq.copy()
        
        # Swap the starting element for this thread's specific branch
        seq[0], seq[first_idx] = seq[first_idx], seq[0]
        
        local_best_seq = seq.copy()
        time_1d = np.zeros(n, dtype=np.int16)
        
        # Score the initial baseline for this branch
        local_best_mse = _process_sequence(tensor, intended, seq, n, time_1d, weighting, t_step_ticks, mode_penalty)
        
        # Generation for the remaining (N-1)! permutations using Heap's algorithm
        M = n - 1
        c = np.zeros(M, dtype=np.int32)
        i = 0
        
        while i < M:
            if c[i] < i:
                if i % 2 == 0:
                    seq[1], seq[1 + i] = seq[1 + i], seq[1]
                else:
                    seq[1 + c[i]], seq[1 + i] = seq[1 + i], seq[1 + c[i]]
                    
                # Evaluate sequence
                mse = _process_sequence(tensor, intended, seq, n, time_1d, weighting, t_step_ticks, mode_penalty)
                
                # Keep if it's the best on the thread
                if mse < local_best_mse:
                    local_best_mse = mse
                    for s in range(n):
                        local_best_seq[s] = seq[s]
                        
                c[i] += 1
                i = 0
            else:
                c[i] = 0
                i += 1
                
        # Save thread's best sequence
        best_mses[first_idx] = local_best_mse
        for s in range(n):
            best_seqs[first_idx, s] = local_best_seq[s]
            
    return best_seqs, best_mses

class SequenceOptimiser:
    
    def __init__(self, env):
        self.env = env
        
        # Checks for mask tensor in in environment engine
        if not hasattr(self.env, 'tensor'):
            raise RuntimeError(
                "Environment Error: Mask Tensor not found. "
                "You must call 'env.calculate_mask_tensor()' before initialising the optimiser."
            )
        
        # Check that spot weights have been set
        if not hasattr(self.env, 'spots_expanded'):
            raise RuntimeError(
                "Environment Error: Spot weights not set."
                "You must call 'env.set_spot_weights()' before initialising the optimiser."
            )

        # Collect and crop the intended distribution the match the evaluation region of the tensor
        self.target_intended = self.env.get_intended_dist()[self.env.tensor_mask]
        
        self.base_sequence = self.env.spots_expanded
        self.seq_len = len(self.base_sequence)
        
        self.t_ticks = np.int16(self.env.t_step_ticks)
        self.mode_pen = np.int16(self.env.mode_pen)
        
        self.rng = np.random.default_rng()
        
    def run(self, method, **kwargs):
        '''
        Unified entry point for all optimisers
        '''
        method = method.lower()
        
        if method == 'montecarlo':
            return self.monte_carlo(**kwargs)
        elif method == 'exhaustive':
            return self.exhaustive(**kwargs)
        elif method == 'greedygd':
            return self.greedy_gd(**kwargs)
        elif method == 'simanneal':
            return self.simulated_annealing(**kwargs)
        elif method == 'mcghybrid':
            return self.mc_greedy_hybrid(**kwargs)
        else:
            raise ValueError(f'Unknown method: {method}')
    
    def monte_carlo(self, n_samples = 10000, weighting=True):
        
        print(f"Running Monte Carlo ({n_samples} samples)")
        start_time = time.perf_counter()
        base_seq = self.base_sequence.astype(np.int16)
        sequences, evals = _numba_mc_core(self.env.tensor, self.target_intended, base_seq, n_samples, weighting, self.t_ticks, self.mode_pen)
        
        best_idx = np.argmin(evals)
        
        duration = time.perf_counter() - start_time
        print(f"Monte Carlo completed in {duration:.4f}s")
        return sequences[best_idx], evals[best_idx], duration
    
    def exhaustive(self, weighting=True):
        if self.seq_len > 12:
            raise MemoryError(f"Sequence length is {self.seq_len}. Exhaustive search requires {factorial(self.seq_len)} permutations, which will take a significant amount of time and may causes crashes. Please use a heuristic method instead.")
        
        print("Running Exhaustive Search")
        start_time = time.perf_counter()

        base_seq = self.base_sequence.astype(np.int16)
        
        # Run exhaustive core
        best_seqs, best_evals = _exhaustive_core(self.env.tensor, self.target_intended, base_seq, weighting, self.t_ticks, self.mode_pen)
        # Find theglobal best sequence
        best_idx = np.argmin(best_evals)
        
        duration = time.perf_counter() - start_time
        print(f"Exhaustive Search completed in {duration:.4f}s")
        
        return best_seqs[best_idx], best_evals[best_idx]
    
    def greedy_gd(self, starting_sequence, iterations=100, weighting=True):
        print("Running Greedy Gradient Descent")
        start_time = time.perf_counter()
        
        current_seq = starting_sequence[np.newaxis, :].astype(np.int16)
        
        # Get initial baseline score
        times = np.zeros(self.seq_len, dtype=np.int16)

        current_mse = _process_sequence(self.env.tensor, self.target_intended, current_seq[0], self.seq_len, times, weighting, self.t_ticks, self.mode_pen)
        
        for i in range(iterations):
            # The Numba core tests all pair swaps in C-memory and returns the absolute best one
            best_neighbor_seqs, best_neighbor_mses = _local_search_core(self.env.tensor, self.target_intended, current_seq, weighting, self.t_ticks, self.mode_pen)
            
            # Did the Numba core find a better sequence?
            if best_neighbor_mses[0] < current_mse:
                current_mse = best_neighbor_mses[0]
                current_seq[0] = best_neighbor_seqs[0]
            else:
                print(f"Converged early at iteration {i}")
                break
                
        duration = time.perf_counter() - start_time
        print(f"Greedy Search completed in {duration:.4f}s")
        return current_seq, current_mse
    
    def simulated_annealing(self, iterations=5000, pop_size=10, temp=0.3, final_temp = 0.001, n_bins = 10, weighting=True, debug = False):
        print("Running Simulated Annealing")
        start_time = time.perf_counter()
        
        # Initialize
        starting_mat = np.tile(self.base_sequence, (pop_size, 1))
        current_seqs = self.rng.permuted(starting_mat, axis = 1)
        
        # Cooling factor
        c_fact = (final_temp / temp) ** (1.0 / iterations)
    
        # Run SA core
        best_seqs, best_mses, all_bins, all_walkers, all_accepted = _sa_core(self.env.tensor, self.target_intended, current_seqs, iterations, temp, c_fact, weighting, n_bins, self.t_ticks, self.mode_pen)
        
        # FInd global minimum
        best_idx = np.argmin(best_mses)
        global_best_seq = best_seqs[best_idx]
        global_best_mse = best_mses[best_idx]
        
        # Average debug trackers
        bin_size = iterations / n_bins
        total_evals = iterations * pop_size
        accepted = np.sum(all_accepted)
        
        # Sum bins across threads and find percentage
        acceptance_bins = np.sum(all_bins, axis=0) / (bin_size * pop_size)
        
        # Average MSE across all walkers
        mse_walker = np.mean(all_walkers, axis=0)
        
        duration = time.perf_counter() - start_time
        print(f"SA completed in {duration:.4f}s (Overall Acceptance of New Solutions: {accepted/total_evals:.1%})")
        
        if debug:
            return global_best_seq, global_best_mse, acceptance_bins, mse_walker, duration
        else:
            return global_best_seq, global_best_mse, duration
      
    def mc_greedy_hybrid(self, n_samples=1000, generations=10, population_size = 10, weighting=True):
        
        print("Running Monte Carlo Greedy Hybrid")
        start_time = time.perf_counter()
        
        base_seq = self.base_sequence.astype(np.int16)
        
        # 1. Get Initial Seed Population
        trial_seqs, trial_evals = _numba_mc_core(self.env.tensor, self.target_intended, base_seq, n_samples, weighting, self.t_ticks, self.mode_pen)
        
        sorted_idx = np.argsort(trial_evals)
        best_seqs = trial_seqs[sorted_idx[:population_size]]
        best_evals = trial_evals[sorted_idx[:population_size]]
        
        # 2. Fire the Mega-Loop
        final_seq, final_mse = _hybrid_generation_loop(self.env.tensor, self.target_intended, base_seq, best_seqs, best_evals, generations, n_samples, weighting, self.t_ticks, self.mode_pen)

        duration = time.perf_counter() - start_time
        print(f"Hybrid Search completed in {duration:.4f}s")
            
        return final_seq, final_mse, duration
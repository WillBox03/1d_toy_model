import numpy as np
from math import factorial, comb
from itertools import combinations
from numba import njit, prange
import time

from ITV_engine import _evaluation_core_single

@njit(parallel=True, cache=True)
def _sa_core(tensor, intended, start_seqs, iterations, temp, c_fact, weighting, n_bins):
    
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
        time_1d = np.zeros(seq_len, dtype=np.int16)
        
        # Find the time of the initial sequence
        time_1d[0] = 0
        for s in range(1, seq_len):
            time_1d[s] = time_1d[s-1] + abs(current_seq[s] - current_seq[s-1])
        
        # Evaluate initial sequence    
        current_mse = _evaluation_core_single(tensor, intended, current_seq, time_1d, weighting)
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
            
            # Recalculate sequence time
            time_1d[0] = 0
            for s in range(1, seq_len):
                time_1d[s] = time_1d[s-1] + abs(test_seq[s] - test_seq[s-1])
            
            # Recalculate error and find difference
            test_mse = _evaluation_core_single(tensor, intended, test_seq, time_1d, weighting)
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
        
        self.base_sequence = self.env.spots_expanded
        self.seq_len = len(self.base_sequence)
        
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

    def _mc_core(self, n_samples, weighting):
        '''
        Generates random sequences and evaluates each one
        '''
        sequences = np.tile(self.base_sequence, (n_samples, 1))
        sequences = self.rng.permuted(sequences, axis = 1)
        
        evals = self.env.evaluate_sequences(sequences, weighting)
        return sequences, evals
        
    def _get_all_pair_swaps(self, sequence):
        """Generates all possible 2-spot swaps for a given sequence
            Works for any number of sequences
        """
        
        if sequence.ndim == 1:
            num_swaps = comb(len(sequence), 2)
            sequences = np.tile(sequence, (num_swaps, 1))
        
            swap_indices = np.array(list(combinations(range(len(sequence)), 2)))
            
            # Vectorized swap
            sequences[np.arange(num_swaps)[:, None], swap_indices] = \
            sequences[np.arange(num_swaps)[:, None], np.flip(swap_indices, axis=-1)]
        
        elif sequence.ndim == 2:
            pop_size, seq_len = sequence.shape
            num_swaps = comb(seq_len, 2)
            
            sequences = np.tile(sequence[:, np.newaxis, :], (1, num_swaps, 1))
            swap_indices = np.array(list(combinations(range(seq_len), 2)))
            
            # Vectorised swap
            col1, col2, = swap_indices[:, 0], swap_indices[:, 1]
            
            temp = sequences[:, np.arange(num_swaps), col1].copy()
            sequences[:, np.arange(num_swaps), col1] = sequences[:, np.arange(num_swaps), col2]
            sequences[:, np.arange(num_swaps), col2] = temp
            
        return sequences
    
    def monte_carlo(self, n_samples = 10000, weighting=True):
        
        print(f"Running Monte Carlo ({n_samples} samples)")
        start_time = time.perf_counter()
        sequences, evals = self._mc_core(n_samples, weighting)
        
        best_idx = np.argmin(evals)
        
        duration = time.perf_counter() - start_time
        print(f"Monte Carlo completed in {duration:.4f}s")
        return sequences[best_idx], evals[best_idx], duration
    
    def exhaustive(self, weighting=True):
        if self.seq_len > 11:
            raise MemoryError(f"Sequence length is {self.seq_len}. Exhaustive search requires {factorial(self.seq_len)} permutations, which will take a significant amount of time and may causes crashes. Please use a heuristic method instead.")
        
        print("Running Exhaustive Search")
        start_time = time.perf_counter()

        # Efficient permutation generation (Steinhaus-Johnson-Trotter logic)
        total_perms = factorial(self.seq_len)
        sequences = np.zeros((total_perms, self.seq_len), np.int16)
        fact = 1
        for m in range(2, self.seq_len + 1):
            frame = sequences[:fact, self.seq_len - m + 1:]
            for i in range(1, m):
                sequences[i * fact : (i + 1) * fact, self.seq_len - m] = i
                sequences[i * fact : (i + 1) * fact, self.seq_len - m + 1:] = frame + (frame >= i)
            frame += 1
            fact *= m
        
        evals = self.env.evaluate_sequences(sequences, weighting)
        best_idx = np.argmin(evals)
        
        duration = time.perf_counter() - start_time
        print(f"Exhaustive Search completed in {duration:.4f}s")
        
        return sequences[best_idx], evals[best_idx]
    
    def greedy_gd(self, starting_sequence, iterations=100, weighting=True):
        print("Running Greedy Gradient Descent")
        start_time = time.perf_counter()

        current_seq = starting_sequence.copy()
        current_mse = self.env.evaluate_sequences(current_seq[np.newaxis, :], weighting)[0]
        
        for i in range(iterations):
            # Generate all neighbours
            neighbours = self._get_all_pair_swaps(current_seq)
            evals = self.env.evaluate_sequences(neighbours, weighting)
            
            best_idx = np.argmin(evals)
            
            if evals[best_idx] < current_mse:
                current_mse = evals[best_idx]
                current_seq = neighbours[best_idx]
                # print(f"Iter {i}: Found better MSE {current_mse:.6f}")
            else:
                print(f"Converged early at iteration {i}")
                break
                
        duration = time.perf_counter() - start_time
        print(f"Greedy Search completed in {duration:.4f}s")
        return current_seq, current_mse
    
    def simulated_annealing(self, iterations=1000, pop_size = 5, temp=1.0, final_temp = 0.001, n_bins = 10, weighting=True, debug = False):
        print("Running Simulated Annealing")
        start_time = time.perf_counter()
        
        # Initialize
        starting_mat = np.tile(self.base_sequence, (pop_size, 1))
        current_seqs = self.rng.permuted(starting_mat, axis = 1)
        
        # Cooling factor
        c_fact = (final_temp / temp) ** (1.0 / iterations)
    
        # Run SA core
        best_seqs, best_mses, all_bins, all_walkers, all_accepted = _sa_core(self.env.tensor, self.env.get_intended_dist(), current_seqs, iterations, temp, c_fact, weighting, n_bins)
        
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
      
    def mc_greedy_hybrid(self, n_samples=1000, generations=10, population_size = 10, weighting=True, comparison_sequence = None, target_improvement=None):
        
        print("Running Monte Carlo Greedy Hybrid")
        start_time = time.perf_counter()
        
        comparison_mse = None
        if comparison_sequence is not None:
            comparison_mse = self.env.evaluate_sequences(comparison_sequence[np.newaxis, :], weighting)[0]
            print(f"Baseline comparison MSE: {comparison_mse:.6f}")
                     
        # Check for early comparison improvement exit conditions               
        def check_target(current_mse):
            if current_mse == 0.0:
                return (comparison_mse - current_mse) / comparison_mse if comparison_mse else 0.0
                
            if comparison_mse is not None and target_improvement is not None:
                imp = (comparison_mse - current_mse) / comparison_mse
                if imp >= target_improvement:
                    return imp
            return False
        
        # Get initial best sequences for monte carlo
        trial_seqs, trial_evals = self._mc_core(n_samples, weighting)
        
        # Sort top 10 sequences
        sorted_idx = np.argsort(trial_evals)
        best_seqs = trial_seqs[sorted_idx[:population_size]]
        best_evals = trial_evals[sorted_idx[:population_size]]
        
        # Check if the initial random population already beats the target improvement
        imp = check_target(best_evals[0])
        if imp is not False:
            duration = time.perf_counter() - start_time
            print(f"Target reached instantly: {imp*100:.2f}% improvement in {duration:.4f}s")
            return best_seqs[0], best_evals[0], duration, imp

        for gen in range(generations):
            # Try to improve locally
            all_neighbours = self._get_all_pair_swaps(best_seqs)
            num_swaps = all_neighbours.shape[1]
            
            # Flatten to evaluate
            all_neighbours_flat = all_neighbours .reshape(-1, self.seq_len)
            evals_flat = self.env.evaluate_sequences(all_neighbours_flat, weighting)
            neighbour_evals = evals_flat.reshape(population_size, num_swaps)
            
            #Find the best of the local updates
            best_neighbour_idx = np.argmin(neighbour_evals, axis = 1)
            best_neighbour_mses = neighbour_evals[np.arange(population_size), best_neighbour_idx]
            
            # Which sequences improved locally
            improved_mask = best_neighbour_mses < best_evals
            
            best_evals[improved_mask] = best_neighbour_mses[improved_mask]
            best_seqs[improved_mask] = all_neighbours[improved_mask, best_neighbour_idx[improved_mask]]
            
            # Which sequences are stagnated and need replacing
            stagnant_mask = ~improved_mask
            stagnant_mask[0] = False # Avoids replacing the global best
            
            num_stagnant = np.sum(stagnant_mask)
            
            if num_stagnant > 0:
                #Generate enough MC samples for the replacement of each stagnant sequence
                n_new_samples = (n_samples // population_size) * num_stagnant
                new_seqs, new_evals = self._mc_core(n_new_samples, weighting)
                    
                # Find the best new relacement sequences
                top_new_idx = np.argsort(new_evals)[:num_stagnant]
                best_seqs[stagnant_mask] = new_seqs[top_new_idx]
                best_evals[stagnant_mask] = new_evals[top_new_idx]
                
            # Check if local swap meets target improvement
            imp = check_target(best_evals[0])
            if imp is not False:
                duration = time.perf_counter() - start_time
                print(f"Target reached during local search: {imp*100:.2f}% improvement in {duration:.4f}s")
                return best_seqs[0], best_evals[0], duration, imp

            # Find new best monte carlo solutions
            new_seqs, new_evals = self._mc_core(n_samples, weighting)
            
            # Replace top 10 sequences with better new monte carlo sequences
            combined_seqs = np.vstack((best_seqs, new_seqs))
            combined_evals = np.concatenate((best_evals, new_evals))
            
            top_idx = np.argsort(combined_evals)[:population_size]
            best_seqs = combined_seqs[top_idx]
            best_evals = combined_evals[top_idx]
            
            # Check target after global population update
            imp = check_target(best_evals[0])
            if imp is not False:
                duration = time.perf_counter() - start_time
                print(f"Target reached after MC injection: {imp*100:.2f}% improvement in {duration:.4f}s")
                return best_seqs[0], best_evals[0], duration, imp

        duration = time.perf_counter() - start_time
        final_improvement = None
        
        if comparison_sequence is not None:
            if comparison_mse != 0:
                final_improvement =  (best_evals[0] - comparison_mse) / comparison_mse
                if final_improvement > 0:
                    print(f"Completed: Sequence improved on comparison by {final_improvement*100:.2f}% in {duration:.4f}s")
                elif final_improvement < 0:
                    print(f"Completed: Sequence is worse than comparison by {-final_improvement*100:.2f}% in {duration:.4f}s")
                else:
                    print(f"Completed: Sequence matches the comparison exactly in {duration:.4f}s")
            else:
                final_improvement = 0.0
                
        else:
            final_improvement = 0.0
                
        print(f"Hybrid Search completed in {duration:.4f}s")
            
        return best_seqs[0], best_evals[0], duration, final_improvement
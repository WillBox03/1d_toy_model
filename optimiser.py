import ITV_engine
import numpy as np
from math import factorial, comb
from itertools import combinations
import time

import sys

np.set_printoptions(threshold=sys.maxsize)

class SequenceOptimiser:
    
    def __init__(self, env):
        self.env = env
        self.n_spots = env.n_spots
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
            return self.greedy_gd(*kwargs)
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
        sequences = self.rng.permuted(np.tile(np.arange(self.n_spots), (n_samples, 1)), axis=1)
        evals = self.env.evaluate_sequences(sequences, weighting)
        return sequences, evals
        
    def _get_all_pair_swaps(self, sequence):
        """Generates all possible 2-spot swaps for a given sequence"""
        
        num_swaps = comb(len(sequence), 2)
        sequences = np.tile(sequence, (num_swaps, 1))
        
        swap_indices = np.array(list(combinations(range(len(sequence)), 2)))
        
        # Vectorized swap
        sequences[np.arange(num_swaps)[:, None], swap_indices] = \
        sequences[np.arange(num_swaps)[:, None], np.flip(swap_indices, axis=-1)]
        
        return sequences
    
    def _get_random_swap(self, sequence):
        """Returns a copy of the sequence with two random spots swapped"""
        seq_copy = sequence.copy()
        idx = range(len(seq_copy))
        i1, i2 = np.random.choice(idx, 2, replace=False)
        seq_copy[i1], seq_copy[i2] = seq_copy[i2], seq_copy[i1]
        return seq_copy
    
    def monte_carlo(self, n_samples = 10000, weighting=True, time_track = False):
        
        print(f"Running Monte Carlo ({n_samples} samples)")
        start_time = time.perf_counter()
        sequences, evals = self._mc_core(n_samples, weighting)
        
        best_idx = np.argmin(evals)
        
        duration = time.perf_counter() - start_time
        print(f"Monte Carlo completed in {duration:.4f}s")
        if not time_track:
            return sequences[best_idx], evals[best_idx]
        else:
            return sequences[best_idx], evals[best_idx], duration
    
    def exhaustive(self, weighting=True):
        print("Running Exhaustive Search")
        start_time = time.perf_counter()

        # Efficient permutation generation (Steinhaus-Johnson-Trotter logic)
        total_perms = factorial(self.n_spots)
        sequences = np.zeros((total_perms, self.n_spots), np.int16)
        fact = 1
        for m in range(2, self.n_spots + 1):
            frame = sequences[:fact, self.n_spots - m + 1:]
            for i in range(1, m):
                sequences[i * fact : (i + 1) * fact, self.n_spots - m] = i
                sequences[i * fact : (i + 1) * fact, self.n_spots - m + 1:] = frame + (frame >= i)
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
    
    def simulated_annealing(self, iterations=1000, temp=1.0, weighting=True, n_bins = 10, final_temp = 0.001, debug = False, time_track = False):
        print("Running Simulated Annealing")
        start_time = time.perf_counter()
        
        # Initialize
        current_seq = np.arange(self.n_spots)
        self.rng.shuffle(current_seq)
        current_mse = self.env.evaluate_sequences(current_seq[np.newaxis, :], weighting)[0]
        
        # Track Global Best
        best_seq = current_seq.copy()
        best_mse = current_mse
        
        T = temp
        accepted = 0
        acceptance_bins = np.zeros((n_bins))
        bin_size = iterations / n_bins
        best_updates = 0
        
        # Calculate the cooling factor
        final_temp = final_temp
        cooling_fact = (final_temp / temp) ** (1 / iterations)
        #print(f'Cooling factor set to {cooling_fact}')
        
        mse_walker = np.zeros(iterations)
        
        for i in range(iterations):
            test_seq = self._get_random_swap(current_seq)
            test_mse = self.env.evaluate_sequences(test_seq[np.newaxis, :], weighting)[0]
            
            diff = test_mse - current_mse
            accepted_step = False
            
            # Acceptance Logic
            if diff < 0:
                current_seq = test_seq
                current_mse = test_mse
                accepted += 1
                accepted_step = True
                
                # Update Global Record
                if current_mse < best_mse:
                    best_mse = current_mse
                    best_seq = current_seq.copy()
                    best_updates += 1
                    #print(f'Global best sequence updated with MSE = {best_mse}, update number {best_updates}, iteration {i}')     
            
            else:
                prob = np.exp(-(diff/current_mse) / T)
                if np.random.rand() < prob:
                    current_seq = test_seq
                    current_mse = test_mse
                    accepted_step = True
                    accepted += 1

            if accepted_step:
                bin_idx = int(min(i // bin_size, n_bins - 1))
                acceptance_bins[bin_idx] += 1
                
            mse_walker[i] = current_mse
                
            T *= cooling_fact
        
        acceptance_bins /= bin_size
        duration = time.perf_counter() - start_time
        print(f"SA completed in {duration:.4f}s (Acceptance: {accepted/iterations:.1%})")
        if debug == True:
            return best_seq, best_mse, acceptance_bins, mse_walker
        elif not time_track:
            return best_seq, best_mse
        else:
            return best_seq, best_mse, duration
    
    def mc_greedy_hybrid(self, n_samples=1000, generations=10, population_size = 10, weighting=True, time_track = False, comparison_sequence = None, time_to_beat = False):
        print("Running Monte Carlo Greedy Hybrid")
        start_time = time.perf_counter()

        # Get initial best sequences for monte carlo
        trial_seqs, trial_evals = self._mc_core(n_samples, weighting)
        
        # Sort top 10 sequences
        sorted_idx = np.argsort(trial_evals)
        best_seqs = trial_seqs[sorted_idx[:population_size]]
        best_evals = trial_evals[sorted_idx[:population_size]]
        
        if comparison_sequence is not None:
            comparison_mse = self.env.evaluate_sequences(comparison_sequence[np.newaxis, :], weighting)[0]
            print(comparison_mse)
            if time_to_beat:
                if best_evals[0] < comparison_mse:
                    duration = time.perf_counter() - start_time
                    print(f"Hybrid Search improved on comparison scan in {duration:.4f}s")
                    return best_seqs[0], best_evals[0], duration
                elif best_evals[0] == 0:
                    duration = time.perf_counter() - start_time
                    print(f"Hybrid Search found a sequence with an error function of 0 in {duration:.4f}s")
                    return best_seqs[0], best_evals[0], duration, -improvement

        for gen in range(generations):
            # Track local updates
            n_improved = 0
            n_replaced = 0
            # Search best sequences locally
            for i in range(population_size):
                # Try to improve locally
                neighbours = self._get_all_pair_swaps(best_seqs[i])
                neighbour_evals = self.env.evaluate_sequences(neighbours, weighting)
                
                #Find the best of the local updates
                best_neighbour_idx = np.argmin(neighbour_evals)
                best_neighbour_mse = neighbour_evals[best_neighbour_idx]
                #Does this improve on the original
                if best_neighbour_mse < best_evals[i]:
                    best_evals[i] = best_neighbour_mse
                    best_seqs[i] = neighbours[best_neighbour_idx]
                    n_improved += 1
                    if time_to_beat:
                        if best_evals[i] < comparison_mse:
                            duration = time.perf_counter() - start_time
                            print(f"Hybrid Search improved on comparison scan in {duration:.4f}s")
                            return best_seqs[0], best_evals[0], duration
                        elif best_evals[i] == 0:
                            duration = time.perf_counter() - start_time
                            print(f"Hybrid Search found a sequence with an error function of 0 in {duration:.4f}s")
                            return best_seqs[0], best_evals[0], duration, -improvement

                #If not, is this the best sequence so far, replace with mc sequence
                elif i > 0:
                    
                    new_seqs, new_evals = self._mc_core(n_samples//population_size, weighting)
                    
                    best_new_idx = np.argmin(new_evals)
                    
                    best_seqs[i] = new_seqs[best_new_idx]
                    best_evals[i] = new_evals[best_new_idx]
                    n_replaced += 1

            # Find new best monte carlo solutions
            new_seqs, new_evals = self._mc_core(n_samples, weighting)
            
            # Replace top 10 sequences with better new monte carlo sequences
            combined_seqs = np.vstack((best_seqs, new_seqs))
            combined_evals = np.concatenate((best_evals, new_evals))
            
            top_idx = np.argsort(combined_evals)[:population_size]
            
            # Track how many new monte carlo sequences are taken in the best sequences
            num_new_entries = np.sum(top_idx >= population_size)
            #print(f"Gen {gen}: Optimised {n_improved} sequences locally, replaced {n_replaced} stagnant sequences, accepted {num_new_entries} new Monte Carlo sequences.")
            
            best_seqs = combined_seqs[top_idx]
            best_evals = combined_evals[top_idx]
            if time_to_beat:
                if best_evals[0] < comparison_mse:
                    duration = time.perf_counter() - start_time
                    print(f"Hybrid Search improved on comparison scan in {duration:.4f}s")
                    return best_seqs[0], best_evals[0], duration
                elif best_evals[0] == 0:
                    duration = time.perf_counter() - start_time
                    print(f"Hybrid Search found a sequence with an error function of 0 in {duration:.4f}s")
                    return best_seqs[0], best_evals[0], duration

        if comparison_sequence is not None:
            improvement =  (best_evals[0] - comparison_mse) / comparison_mse
            if improvement < 0:
                print(f'The Hybrid Search sequence improves on the comparison sequence by {-improvement*100:.2f}%')
            elif improvement > 0:
                print(f'The Hybrid Search sequence is worse than the comparison sequence by {improvement*100:.2f}%')
            else:
                print('The Hybrid search has found a sequence with error function equal to the comparison sequence')
        
        duration = time.perf_counter() - start_time
        print(f"Hybrid Search completed in {duration:.4f}s")
        if not time_track:
            return best_seqs[0], best_evals[0]
        else:
            return best_seqs[0], best_evals[0], duration, -improvement
        
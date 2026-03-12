import numpy as np
import math
import matplotlib.pyplot as plt
from numba import njit, prange

@njit(parallel=True, cache = True)
def _evaluation_core(tensor, intended, sequences, time_indices, weighting):
    
    n_sequences = sequences.shape[0]
    n_phases = tensor.shape[0]
    n_voxels = tensor.shape[3]
    seq_len = sequences.shape[1]
    
    evals = np.zeros(n_sequences)
    
    for i in prange(n_sequences):
        total_sq_diff = 0.0
        
        # Hold dosage for this phase
        phase_dose = np.zeros(n_voxels)
        for p in range(n_phases):
            phase_dose = np.zeros(n_voxels)
            
            # For each sequence spot
            for s in range(seq_len):
                spot = sequences[i, s]
                time_idx = time_indices[i, s]
                
                # For each voxel
                for v in range(n_voxels):
                    phase_dose[v] += tensor[p, spot, time_idx, v]
            
            # Calculate MSE for the phase
            for v in range(n_voxels):
                diff = phase_dose[v] - intended[v]
                sq_diff = diff * diff
                
                if weighting and diff < 0:
                    sq_diff *= 5.0
                    
                total_sq_diff += sq_diff
                
        # Average
        evals[i] = total_sq_diff / (n_phases * n_voxels)
        
    return evals

@njit(cache = True)
def _evaluation_core_single_thread(tensor, intended, seq, time_indices, weighting):
    
    n_phases = tensor.shape[0]
    n_voxels = tensor.shape[3]
    seq_len = seq.shape[0]
    
    total_sq_diff = 0.0
    phase_dose = np.zeros(n_voxels) # Allocate once!
    
    for p in range(n_phases):
        # Zero out the dose array
        for v in range(n_voxels):
            phase_dose[v] = 0.0
            
        # Accumulate dose (Fast contiguous memory read)
        for s in range(seq_len):
            spot = seq[s]
            time_idx = time_indices[s]
            for v in range(n_voxels):
                phase_dose[v] += tensor[p, spot, time_idx, v]
                
        # Calculate Error
        for v in range(n_voxels):
            diff = phase_dose[v] - intended[v]
            sq_diff = diff * diff
            if weighting and diff < 0:
                sq_diff *= 5.0
            total_sq_diff += sq_diff
            
    return total_sq_diff / (n_phases * n_voxels)
class ITV_env:   
    '''
    Environment space for the simulation
    '''
    def __init__(self, res, ctv_length, n_spots, amp, spot_size = None, t_step = 0.1, t_switch = 0.5, margin = None):
        '''
        Initialises the environment, creating a high resolution coordinate space for the env of set length. Also defines positions of proton beam spots covering the ITV
        Args:
            res (float): Resolution of environment, in cm
            ctv_length (float,int): Length of env, in cm
            n_spots (int): Number of proton beam spots covering the ITV
            amp(float): Amplitude of breathing motion, in cm
            spot_size (float): Physical width of spot in cm
        '''
        
        if spot_size is None:
            spot_size = (ctv_length + (2*amp)) / n_spots
            
        if margin is None:
            margin = amp
        
        self.update_env(res=res, ctv_length=ctv_length, amp=amp, n_spots=n_spots, spot_size=spot_size, t_step=t_step, t_switch=t_switch, margin=margin)
        
        # Convert seconds to integers
        multiplier = 100000 # Correct for floating point errors
        int_step = int(np.round(self.t_step * multiplier))
        int_switch = int(np.round(self.t_switch * multiplier))
        
        # Find the greatest common divisor to get the engine tick time
        gcd_val = math.gcd(int_step, int_switch)
        self.tick_time = gcd_val / multiplier
        
        # Calculate number of ticks for stepping and energy switching
        self.t_step_ticks = int(np.round(self.t_step / self.tick_time))
        self.t_switch_ticks = int(np.round(self.t_switch / self.tick_time))
        
        self.mode_presets = {
            '1d': 0,
            'q2dslice': self.t_step_ticks, 
            'q2dlayer': self.t_switch_ticks
        } 
                
    def update_env(self, **kwargs):
        '''
        Update the number of proton spots and the oscillation amplitude without initialising a new object
        '''
        
        if 'res' in kwargs: self.res = kwargs['res']
        if 'ctv_length' in kwargs: self.ctv_length = kwargs['ctv_length']
        if 'amp' in kwargs: self.amp = kwargs['amp']
        if 'n_spots' in kwargs: self.n_spots = kwargs['n_spots']
        if 'spot_size' in kwargs: self.spot_size = kwargs['spot_size']
        if 't_step' in kwargs: self.t_step = kwargs['t_step']
        if 't_switch' in kwargs: self.t_switch = kwargs['t_switch']
        if 'margin' in kwargs: self.margin = kwargs['margin']
        
        # Boundaries of env and ITV
        self.ctv_half_length = self.ctv_length / 2
        self.itv_extent = self.ctv_half_length + self.amp
        
        # Extent of environment, including a margin around the ITV
        self.total_env_half_length = self.itv_extent + self.margin
        self.total_env_length = 2 * self.total_env_half_length
        
        self.n_env_voxels = int(np.round(self.total_env_length / self.res)) # Calculate number of voxels used in env
        
        # Create hi-res env environment at voxel centres
        self.env_space = np.linspace(-self.total_env_half_length + self.res/2, self.total_env_half_length - self.res/2, self.n_env_voxels)
        
        total_itv_width = 2 * self.itv_extent
        total_spot_width = self.n_spots * self.spot_size
        
        if total_spot_width > total_itv_width:
            new_spot_size = total_itv_width / self.n_spots
            print(f"WARNING: {self.n_spots} spots of size {self.spot_size}cm cannot fit in an ITV of {total_itv_width}cm without spot overlap. Spot width set to {new_spot_size}")
            self.spot_size = new_spot_size
        
        # Define spot centres and edges
        spot_centres = np.linspace(-self.itv_extent + self.spot_size/2, self.itv_extent - self.spot_size/2, self.n_spots)
        
        # Define proton spot edges in environment
        self.spot_left_edges = spot_centres - self.spot_size / 2
        self.spot_right_edges = spot_centres + self.spot_size / 2

        # Define empty attributes for spot weights and dose distribution
        self.spot_weights = np.zeros(self.n_spots, dtype=int)
        self.passes = 1
        self.spots_expanded = np.arange(self.n_spots)

    def set_spot_weights(self, spot_weights, repaints = 0, bias = 3):
        '''
        Defines the dosage weighting for each spot
        '''
        self.passes = repaints + 1
        
        presets = {
            'uniform': np.ones(self.n_spots, dtype=int),
            # Wedges, deposits more dose to one side of the env
            'wedge_lr': np.round(np.linspace(1, bias, self.n_spots)).astype(int),
            'wedge_lr': np.round(np.linspace(3, bias, self.n_spots)).astype(int),
            # Increased dose in the centre of the env
            'centre_peak': np.round(1 + (bias - 1) * (1 - np.abs(np.linspace(-1, 1, self.n_spots)))).astype(int),
            # Increased dose at the edges of the env
            'edges_peak': np.round(1 + (bias - 1) * np.abs(np.linspace(-1, 1, self.n_spots))).astype(int)
        }

        if isinstance(spot_weights, str):
            if spot_weights in presets:
                self.spot_weights = presets[spot_weights]
            else:
                raise ValueError(f"Unknown weighting preset: '{spot_weights}'. Available: {list(presets.keys())}")
        else:
            self.spot_weights = np.array(spot_weights, dtype=int)
            if len(self.spot_weights) != self.n_spots:
                raise ValueError(f'Custom weights array length ({len(self.spot_weights)}) does not match the number of beam spots ({self.n_spots}).')
            
        self.spot_weights *= self.passes
        self.spots_expanded = np.repeat(np.arange(self.n_spots), self.spot_weights)

    def set_sequence(self, sequence, bidirectional = True):
        '''
        Defines the ordering sequence of the spots, with some preset order selections
        '''
        
        def raster_repaint(bidirectional = True):
            
            single_raster = np.arange(self.n_spots)
            
            if not bidirectional:
                return np.tile(single_raster, self.passes)
            
            sequence = [single_raster if i % 2 == 0 else single_raster[::-1] for i in range(self.passes)]
            return np.concatenate(sequence)
            
        def find_max_dist():
            
            A = np.sort(self.spots_expanded)
            N = len(A)
            half = N // 2
            
            if N % 2 == 0:
                lower = A[:half]
                upper = A[half:][::-1] 
                
                seq = np.empty(N, dtype=A.dtype)
                seq[0::2] = lower
                seq[1::2] = upper
                
                return np.roll(seq, 1)
            else:
                lower = A[:half]
                upper = A[half+1:][::-1]
                mid = A[half]
                
                seq = np.empty(N-1, dtype=A.dtype)
                seq[0::2] = lower
                seq[1::2] = upper
                
                return np.concatenate((np.array([mid]), np.roll(seq, 1)))
            
        presets = {'lr_rast' : raster_repaint(bidirectional),
                    'rl_rast' : raster_repaint(bidirectional)[::-1],
                    'rand' : np.random.permutation(self.spots_expanded),
                    'max_dist': find_max_dist()
        }
        
        if isinstance(sequence, str):
            if sequence in presets:
                return presets[sequence]
            else:
                raise ValueError(f"Unknown preset: '{sequence}'. Available: {list(presets.keys())}")  

        return sequence

    def sim(self, period, sequence, starting_phase = None, phases = 8, weighting = True, mode = '1d', region = 'ctv'):
        
        '''
        Simulates spot scanning sequence in a moving target, returning the resultant dose distribution
        Args:
            t_step (float): Time taken for proton beam to move to adjacent spots, in s
            period (float): Period of breathing motion, in s
            sequence (arr, str): Sequence for proton beam spots
        '''
        
        if mode not in self.mode_presets:
            raise ValueError(f'Mode "{mode}" not recognised. Available: {list(self.mode_presets.keys())}')
        
        sequence = self.set_sequence(sequence)
        
        phase = 2*np.pi/period # Phase of breathing
        
        jumps = abs(sequence[1:] - sequence[:-1]) # Distance between each proton beam spot in sequence

        ticks = jumps * self.t_step_ticks
        
        # Apply mode penalty to 0 jumps
        ticks[jumps == 0] = self.mode_presets.get(mode, 0)
        
        # Accumulate the integers ticks
        cumulative_ticks = np.zeros(len(jumps) + 1, dtype=int)
        cumulative_ticks[1:] = np.cumsum(ticks)
        
        # Convert ticks to seconds for physical calculations
        times = cumulative_ticks * self.tick_time
        
        if starting_phase is None:
            # Vectorize over selected phases
            phases = np.linspace(0, 2*np.pi, phases, endpoint=False)[:, np.newaxis]
        else:
            # Simulate just the single specified phase
            phases = np.array([starting_phase])[:, np.newaxis]

        shifts = self.amp*np.sin((phase*times) + phases) # Resultant positional shift at each time

        # Find the spot edges in sequence order
        left_edges = self.spot_left_edges[sequence]
        right_edges = self.spot_right_edges[sequence]

        # Find spot edges relative to env position due to shifts  
        left_shifts = (left_edges - shifts)[:, :, np.newaxis]
        right_shifts = (right_edges - shifts)[:, :, np.newaxis]
        
        # Mask for proton spots over the env
        spot_masks = (self.env_space > left_shifts) & (self.env_space < right_shifts)

        # Add dose of 1 at every part of env space visited by a proton spot
        dose_dist = np.sum(spot_masks, axis = 1)
        
        dose_dist = dose_dist if starting_phase is None else dose_dist[0]
        
        error = self.calc_error(dose_dist, weighting = weighting, region = region)

        return dose_dist, error   

    def get_intended_dist(self):
        '''
        Deliver the intended distribution to env
        '''
        left_edges = self.spot_left_edges[:, np.newaxis]
        right_edges = self.spot_right_edges[:, np.newaxis]
    
        spot_masks = (self.env_space > left_edges) & (self.env_space < right_edges)
        
        return spot_masks.T @ self.spot_weights
    
    def calc_error(self, dose_dist, weighting = True, region = 'ctv'):
        '''
        Calculates the error of a resultant dose distribution against the intended distribution.
        If weighting=True, applies a penalty (x5) to underdosed voxels (WMSE).
        Otherwise, returns standard MSE.
        '''
        
        # Check which region is evaluated and mask:
        if region == 'ctv':
            region_mask = (self.env_space >= -self.ctv_half_length) & (self.env_space <= self.ctv_half_length)
        elif region == 'itv':
            region_mask = (self.env_space >= -self.itv_extent) & (self.env_space <= self.itv_extent)
        else:
            raise ValueError("Region must be 'ctv' or 'itv'")
        
        # Check for starting phase and applies mask
        if dose_dist.ndim == 1:
            region_dose = dose_dist[region_mask]
        else:
            region_dose = dose_dist[:, region_mask]
        
        intended_dist = self.get_intended_dist()
        
        intended_dose= intended_dist[region_mask]
        
        # Calculate raw differences
        diffs = region_dose - intended_dose
        sq_diffs = diffs**2
        
        # Apply penalty to negative differences (underdosing) if weighting is on
        if weighting:
            sq_diffs[diffs < 0] *= 5
            
        return np.mean(sq_diffs)
          
    def calculate_mask_tensor(self, period, starting_phase = None, phases = 8, mode = '1d', region = 'ctv'):
        """
        Precomputes all possible spot/time combinations.
    
        Args:
            timestep (float): Time per jump unit.
            period (float): Breathing period.
            starting_phase (float/None): 
                If float: returns 3D Tensor (Spots, Time, Voxels)
                If None: returns 4D Tensor (All Phases, Spots, Time, Voxels)
        """
        
        phase_speed = 2*np.pi/period # Phase of breathing
        
        if mode not in self.mode_presets:
            raise ValueError(f'Mode "{mode}" not recognised. Available: {list(self.mode_presets.keys())}')
        
        self.mode_pen = self.mode_presets.get(mode, 0)
        
        # Convert max jump distance to ticks
        max_jump_ticks = (self.n_spots - 1) * self.t_step_ticks
        
        # Compare longest jump to energy layer switch
        worst_jump_ticks = max(max_jump_ticks, self.mode_pen)
        
        # Calculate max possible time in ticks
        spot_visits = np.sum(self.spot_weights)
        t_max_ticks = int(max(0, (spot_visits - 1) * worst_jump_ticks))
        
        # Convert ticks to seconds for generation physics
        times = (np.arange(t_max_ticks + 1) * self.tick_time)[np.newaxis, :]
    
        # Determine phase list from start_phase argument
        if starting_phase is None:
            # Use standard number of phases
            phases = np.linspace(0, 2*np.pi, phases, endpoint=False)[:, np.newaxis]
        else:
            phases = np.array([starting_phase])[:, np.newaxis]
            
        #Create 4D tensor, currently filled with the phase shifts at each timepoint with each starting phase
        #Dimensions (P,1,T,1)
        all_shifts = (self.amp*np.sin((phase_speed*times) + phases))[:, np.newaxis, :, np.newaxis]

        #Find the left and right edges in dimensions (1, S, 1, 1)
        left_edges = self.spot_left_edges[np.newaxis, :, np.newaxis, np.newaxis]
        right_edges = self.spot_right_edges[np.newaxis, :, np.newaxis, np.newaxis]
        
        # Define mask based on evaluation region
        if region.lower() == 'ctv':
            self.tensor_mask = (self.env_space >= -self.ctv_half_length) & (self.env_space <= self.ctv_half_length)
        elif region.lower() == 'itv':
            self.tensor_mask = (self.env_space >= -self.itv_extent) & (self.env_space <= self.itv_extent)
        else:
            raise ValueError("Region must be 'ctv' or 'itv'")
        
        target_env_space = self.env_space[self.tensor_mask]
        
        #Reshape env space to (1,1,1,V)
        env_space = target_env_space[np.newaxis, np.newaxis, np.newaxis, :]

        #Generate the masks for all phases, for all spots, for all time steps
        #Resulting tensor has dimensions (P,S,T,V)
        self.tensor = (env_space > (left_edges - all_shifts)) & (env_space < (right_edges - all_shifts))
    
    def evaluate_sequences(self, sequences, weighting = True):
        """
        Calculates MSE for a batch of sequences.
        Args:
            sequences: Array of sample sequences
            weighting(bool): Determines whether weighted mse or regular mse is used for evaluation
        """
        # Check if only one sequence is passed in
        if sequences.ndim == 1:
            sequences = sequences[np.newaxis, :]
        
        # Calculate physical jump distances
        jumps = np.abs(np.diff(sequences, axis=1))

        # Convert to ticks
        ticks = jumps * self.t_step_ticks

        # Apply mode penalty
        ticks[jumps == 0] = self.mode_pen

        # Finding the cumulative jump distances as tensor indices
        time_indices = np.zeros(sequences.shape, dtype=int)
        time_indices[:, 1:] = np.cumsum(ticks, axis=1)
        
        intended_mask = self.get_intended_dist()[self.tensor_mask]
        
        return _evaluation_core(self.tensor, intended_mask, sequences, time_indices, weighting)     
    
    def display(self, sims, labels = None, avg_phases = False):
        '''
        Generate and display the intended target distribution, along with other simulated distributions
        '''
        
        target_dist = self.get_intended_dist()
        
        ctv_intended = np.zeros_like(self.env_space)
        ctv_mask = (self.env_space >= -self.ctv_half_length) & (self.env_space <= self.ctv_half_length)
        ctv_intended[ctv_mask] = np.max(target_dist)
                
        for i, sim in enumerate(sims):
            #Assign label if it exists
            if labels is not None and i < len(labels):
                sim_label = labels[i]
            else:
                sim_label = f'Simulated Distribution {i+1}'
            if sim.ndim == 1:
                plt.plot(self.env_space, sim, lw = 2, alpha = 0.9, label = sim_label)
            else:
                if avg_phases:
                    # Average phases distributions if needed
                    avg_dist = np.mean(sim, axis = 0)
                    plt.plot(self.env_space, avg_dist, lw = 2, alpha = 0.9, label = sim_label)
                else:
                    for j, dist in enumerate(sim):
                        current_label = sim_label if j == 0 else '_nolegend_'
                        
                        plt.plot(self.env_space, dist, lw = 2, alpha = 0.9, label = current_label)
            
        plt.plot(self.env_space, target_dist, 'k--', lw=2, label = "Intended Dose (ITV)")
        plt.plot(self.env_space, ctv_intended, 'r:', lw=2, label = "CTV")   
            
        plt.ylabel('Relative Dose (Times Visited)')
        plt.xlabel('Position on env (cm)')
        plt.grid(True)
        
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True)
        plt.tight_layout()
        plt.show()
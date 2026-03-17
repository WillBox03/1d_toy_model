import numpy as np
import math
import matplotlib.pyplot as plt
from numba import njit, prange

@njit(parallel=True, cache=True)
def _build_tensor(shifts, spot_l, spot_r, env, max_hits):
    
    P = shifts.shape[0] # Starting Phases
    T = shifts.shape[1] # Time points
    S = len(spot_l) # Number of spots
    V = len(env) # Number of voxels in environment
    
    # Initialise tensor with -1 for empty voxels
    tensor = np.full((P, S, T, max_hits), -1, dtype=np.int32)
    
    for p in prange(P):
        for s in range(S):
            for t in range(T):
                hit_count = 0
                shift = shifts[p, t]
                
                edge_l = spot_l[s] - shift
                edge_r = spot_r[s] - shift
                
                for v in range(V):
                    if env[v] > edge_l and env[v] < edge_r:
                        if hit_count < max_hits:
                            tensor[p, s, t, hit_count] = v
                            hit_count += 1
    
    return tensor

@njit(parallel=True, cache=True)
def _build_tensor_2d(shifts_x, shifts_y, spot_l, spot_r, spot_b, spot_t, env_x, env_y, max_hits):
    
    P = shifts_x.shape[0] # Starting Phases
    T = shifts_x.shape[1] # Time points
    S = len(spot_l) # Number of spots
    V = len(env_x) # Number of voxels in environment
    
    # Initialise tensor with -1 for empty voxels
    tensor = np.full((P, S, T, max_hits), -1, dtype=np.int32)
    
    for p in prange(P):
        for s in range(S):
            for t in range(T):
                hit_count = 0
                shift_x = shifts_x[p, t]
                shift_y = shifts_y[p, t]
                
                edge_l = spot_l[s] - shift_x
                edge_r = spot_r[s] - shift_x
                edge_b = spot_b[s] - shift_y
                edge_t = spot_t[s] - shift_y
                
                for v in range(V):
                    if env_x[v] > edge_l and env_x[v] < edge_r and env_y[v] > edge_b and env_y[v] < edge_t:
                        if hit_count < max_hits:
                            tensor[p, s, t, hit_count] = v
                            hit_count += 1
    
    return tensor

@njit(cache = True)
def _evaluation_core_single_thread(tensor, intended, seq, time_indices, weighting):
    
    n_phases = tensor.shape[0]
    n_voxels = len(intended)
    max_hits = tensor.shape[3]
    seq_len = seq.shape[0]
    
    total_sq_diff = 0.0
    phase_dose = np.zeros(n_voxels)
    
    for p in range(n_phases):
        # Zero out the dose array
        for v in range(n_voxels):
            phase_dose[v] = 0.0
            
        # Accumulate dose
        for s in range(seq_len):
            spot = seq[s]
            time_idx = time_indices[s]
            # Search voxels which are hit with dose
            for h in range(max_hits):
                v_idx = tensor[p, spot, time_idx, h] # Get voxel environment index
                
                if v_idx == -1: # If index is -1, exit loop
                    break
                
                phase_dose[v_idx] += 1.0
                
        # Calculate Error
        for v in range(n_voxels):
            diff = phase_dose[v] - intended[v]
            sq_diff = diff * diff
            if weighting and diff < 0:
                sq_diff *= 5.0
            total_sq_diff += sq_diff
            
    return total_sq_diff / (n_phases * n_voxels) # Return the average MSE

@njit(parallel=True, cache=True)
def _evaluation_core(tensor, intended, sequences, time_indices, weighting):
    
    n_sequences = sequences.shape[0]
    n_phases = tensor.shape[0]
    max_hits = tensor.shape[3]
    n_voxels = len(intended)
    seq_len = sequences.shape[1]
    
    evals = np.zeros(n_sequences)
    
    for i in prange(n_sequences):
        total_sq_diff = 0.0
        
        for p in range(n_phases):
            phase_dose = np.zeros(n_voxels)
            
            # For each sequence spot
            for s in range(seq_len):
                spot = sequences[i, s]
                time_idx = time_indices[i, s]
                
                # Search voxels which are hit with dose
                for h in range(max_hits):
                    v_idx = tensor[p, spot, time_idx, h] # Get voxel environment index
                    
                    if v_idx == -1: # If index is -1, exit loop
                        break
                    
                    phase_dose[v_idx] += 1.0
            
            # Calculate MSE for the phase
            for v in range(n_voxels):
                diff = phase_dose[v] - intended[v]
                sq_diff = diff * diff
                
                if weighting and diff < 0:
                    sq_diff *= 5.0
                    
                total_sq_diff += sq_diff
                
        # Average the MSE
        evals[i] = total_sq_diff / (n_phases * n_voxels)
        
    return evals # Return all MSEs

def find_max_dist(spots_expanded):
            '''
            Returns the maxiumum distance/time sequence
            Starts with the centre spot, then jumps to the furthest spot away until all spots have been met
            '''
            A = np.sort(spots_expanded) # Sort spot positions
            N = len(A)
            half = N // 2
            
            if N % 2 == 0:
                # If number of spots is even, split into equal arrays
                lower = A[:half]
                upper = A[half:][::-1] 
                
                # Interleave sequence with spots from alternating arrays
                seq = np.empty(N, dtype=A.dtype)
                seq[0::2] = lower
                seq[1::2] = upper
                
                return np.roll(seq, 1) # Start with the middle spot
            else:
                # If number of spots is odd, extract centre spot
                lower = A[:half]
                upper = A[half+1:][::-1]
                mid = A[half]
                
                seq = np.empty(N-1, dtype=A.dtype)
                seq[0::2] = lower
                seq[1::2] = upper
                
                return np.concatenate((np.array([mid]), np.roll(seq, 1))) # Add spot to the front at the end
            
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
            
        presets = {'lr_rast' : raster_repaint(bidirectional),
                    'rl_rast' : raster_repaint(bidirectional)[::-1],
                    'rand' : np.random.permutation(self.spots_expanded),
                    'max_dist': find_max_dist(self.spots_expanded)
        }
        
        if isinstance(sequence, str):
            if sequence in presets:
                return presets[sequence]
            else:
                raise ValueError(f"Unknown preset: '{sequence}'. Available: {list(presets.keys())}")  

        return sequence

    def sim(self, period, sequence, starting_phase = None, n_phases = 24, weighting = True, mode = '1d', region = 'ctv'):
        
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
            phases = np.linspace(0, 2*np.pi, n_phases, endpoint=False)[:, np.newaxis]
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
        # Check for starting phase and apply mask
        region_dose = dose_dist[region_mask] if dose_dist.ndim == 1 else dose_dist[:, region_mask]
        
        intended_dist = self.get_intended_dist()
        
        intended_dose= intended_dist[region_mask]
        
        # Calculate raw differences
        diffs = region_dose - intended_dose
        sq_diffs = diffs**2
        
        # Apply penalty to negative differences (underdosing) if weighting is on
        if weighting:
            sq_diffs[diffs < 0] *= 5
            
        return np.mean(sq_diffs)
          
    def calculate_mask_tensor(self, period, starting_phase = None, n_phases = 24, mode = '1d', region = 'ctv'):
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
            phases = np.linspace(0, 2*np.pi, n_phases, endpoint=False)[:, np.newaxis]
        else:
            phases = np.array([starting_phase])[:, np.newaxis]
            
        # Create shift matrix
        shifts = self.amp*np.sin((phase_speed*times) + phases)
        
        # Define target region
        if region.lower() == 'ctv':
            self.tensor_mask = (self.env_space >= -self.ctv_half_length) & (self.env_space <= self.ctv_half_length)
        elif region.lower() == 'itv':
            self.tensor_mask = (self.env_space >= -self.itv_extent) & (self.env_space <= self.itv_extent)
        else:
            raise ValueError("Region must be 'ctv' or 'itv'")
        
        # Target voxel coords
        target_env = self.env_space[self.tensor_mask]
        
        # Calculate the maximum possible number of voxels hit by a spot
        max_hits = int(np.ceil(self.spot_size / self.res)) + 2

        # Generate tensor for all phases, for all spots, for all time steps, for all the voxels hit
        self.tensor = _build_tensor(shifts, self.spot_left_edges, self.spot_right_edges, target_env, max_hits)
    
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
    
    def display(self, sims = None, labels = None, avg_phases = False, spot_edges = False):
        '''
        Generate and display the intended target distribution, along with other simulated distributions
        '''
        
        target_dist = self.get_intended_dist()
        
        ctv_intended = np.zeros_like(self.env_space)
        ctv_mask = (self.env_space >= -self.ctv_half_length) & (self.env_space <= self.ctv_half_length)
        ctv_intended[ctv_mask] = np.max(target_dist)
        
        if sims is not None:
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
        
        if spot_edges:
            plt.axvline(self.spot_left_edges[0], color='blue', linestyle='--', linewidth=1, label='Spot Edges')
            plt.axvline(self.spot_right_edges[0], color='blue', linestyle='--', linewidth=1)
            for i in range(1, len(self.spot_left_edges)):
                plt.axvline(self.spot_left_edges[i], color='blue', linestyle='--', linewidth=1)
                plt.axvline(self.spot_right_edges[i], color='blue', linestyle='--', linewidth=1)
        
        plt.title('Dose Distribution in 1D')
        plt.ylabel('Relative Dose (Times Visited)')
        plt.xlabel('Position on env (cm)')
        plt.grid(True)
        
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True)
        plt.tight_layout()
        plt.show()
        
        
        
        

class ITV_env_2D:
    '''
    Environment space for the simulation in 2D
    '''
    def __init__(self, ctv_size, n_spots, amp, res = 0.1 , spot_size = None, t_steps = (0.1,0.1), margin = None):
        '''
        Initialises the environment, creating a high resolution coordinate space for the env of set length. Also defines positions of proton beam spots covering the ITV
        Args:
            res (float): Resolution of environment, in cm
            ctv_size (tuple, float): Dimensions of CTV in x and y, in cm
            n_spots (tuple, int): Number of proton beam spots covering the ITV in x and y dims
            amp(float): Amplitude of breathing motion in both x and y dims, in cm
            spot_size (float, float/None): Physical width of spot in cm
            t_steps (tuple, float): Time taken for proton beam to move to adjacent spots, in x and y dims, in s
            margin(tuple, float): Space between edge of ITV and edge of simulation in x and y dims
        '''
        
        if spot_size is None:
            spot_size = tuple((np.array(ctv_size) + (2 * np.array(amp))) / np.array(n_spots))
            
        if margin is None:
            margin = amp
        
        self.update_env(res=res, ctv_size=ctv_size, amp=amp, n_spots=n_spots, spot_size=spot_size, t_steps=t_steps, margin=margin)
                
    def update_env(self, **kwargs):
        '''
        Update the number of proton spots and the oscillation amplitude without initialising a new object
        '''
        
        if 'res' in kwargs: self.res = kwargs['res']
        if 'ctv_size' in kwargs: self.ctv_size = kwargs['ctv_size']
        if 'amp' in kwargs: self.amp = kwargs['amp']
        if 'n_spots' in kwargs: self.n_spots = kwargs['n_spots']
        if 'spot_size' in kwargs: self.spot_size = kwargs['spot_size']
        if 't_steps' in kwargs: self.t_steps = kwargs['t_steps']
        if 'margin' in kwargs: self.margin = kwargs['margin']
        
        self.n_spots_tot = self.n_spots[0] * self.n_spots[1] # Total spot number
        
        # Unpack CTV size, amplitude and margins
        self.ctv_half_x, self.ctv_half_y = self.ctv_size[0] / 2, self.ctv_size[1] / 2
        self.amp_x, self.amp_y = self.amp
        self.margin_x, self.margin_y = self.margin
        
        # Boundaries of ITV, and lengths of the environment
        self.itv_extent_x = self.ctv_half_x + self.amp_x
        self.itv_extent_y = self.ctv_half_y + self.amp_y
        
        self.total_env_half_x = self.itv_extent_x + self.margin_x
        self.total_env_half_y = self.itv_extent_y + self.margin_y
        
        # Determine environment grid size
        self.n_voxels_x = int(np.round((2 * self.total_env_half_x) / self.res))
        self.n_voxels_y = int(np.round((2 * self.total_env_half_y) / self.res))
        
        # Create 2D coordinate space
        x_axis = np.linspace(-self.total_env_half_x + self.res/2, self.total_env_half_x - self.res/2, self.n_voxels_x)
        y_axis = np.linspace(-self.total_env_half_y + self.res/2, self.total_env_half_y - self.res/2, self.n_voxels_y)
        X, Y = np.meshgrid(x_axis, y_axis)        
        
        # Flatten for calculations
        self.env_space_x = X.flatten()
        self.env_space_y = Y.flatten()
        self.n_env_voxels = len(self.env_space_x) # Number of voxels in env
        
        # Unpack spot centres,m list and flatten into arrays
        spot_size_x, spot_size_y = self.spot_size
        spot_axis_x = np.linspace(-self.itv_extent_x + spot_size_x/2, self.itv_extent_x - spot_size_x/2, self.n_spots[0])
        spot_axis_y = np.linspace(-self.itv_extent_y + spot_size_y/2, self.itv_extent_y - spot_size_y/2, self.n_spots[1])
        SX, SY = np.meshgrid(spot_axis_x, spot_axis_y)
        self.spot_centres_x = SX.flatten()
        self.spot_centres_y = SY.flatten()
        
        # Define proton spot edges in environment
        self.spot_left = self.spot_centres_x - spot_size_x / 2
        self.spot_right = self.spot_centres_x + spot_size_x / 2
        self.spot_bottom = self.spot_centres_y - spot_size_y / 2
        self.spot_top = self.spot_centres_y + spot_size_y / 2
        
        # Calculating engine time constants
        multiplier = 100000 # Correct for floating point errors
        int_step_x = int(np.round(self.t_steps[0] * multiplier))
        int_step_y = int(np.round(self.t_steps[1] * multiplier))
        
        # Find the greatest common divisor to get the engine tick time
        gcd_val = math.gcd(int_step_x, int_step_y)
        self.tick_time = gcd_val / multiplier
        
        # Update tick counts
        self.t_step_ticks_x = int(np.round(self.t_steps[0] / self.tick_time))
        self.t_step_ticks_y = int(np.round(self.t_steps[1] / self.tick_time))

        # Define empty attributes for spot weights and dose distribution
        self.spot_weights = np.zeros(self.n_spots_tot, dtype=int)
        self.passes = 1
        self.spots_expanded = np.arange(self.n_spots_tot)

    def set_spot_weights(self, spot_weights, repaints = 0):
        '''
        Defines the dosage weighting for each spot, only accepts uniform as preset currently
        '''
        self.passes = repaints + 1
        
        self.passes = repaints + 1
        if spot_weights == 'uniform':
            self.spot_weights = np.ones(self.n_spots_tot, dtype=int)
        elif len(spot_weights) != self.n_spots_tot:
            raise ValueError(f'Length of custom spot_weights argument ({len(spot_weights)}), does not equal the number of spots ({self.n_spots_tot}).')
        else:
            self.spot_weights = np.array(spot_weights, dtype=int)
            
        self.spot_weights *= self.passes
        self.spots_expanded = np.repeat(np.arange(self.n_spots_tot), self.spot_weights)
    
    def set_sequence(self, sequence = 'raster', bidirectional = False):
        '''
        Defines the ordering sequence of the spots, with some preset order selections
        '''
        
        def raster_repaint(bidirectional = True):
            
            nx = self.n_spots[0]
            ny = self.n_spots[1]
            
            # Create a 2D grid of the spot numbers
            grid = np.arange(self.n_spots_tot).reshape(ny, nx)
            
            # Flip every second row
            for i in range(1, ny, 2):
                grid[i] = grid[i, ::-1]
            
            # Flatten it back to 1D
            single_rast = grid.flatten()
            
            # If bidirectional is False,m repeat the same snake [0...99, 0...99]
            if not bidirectional:
                return np.tile(single_rast, self.passes)
            
            # Otherwise, paint back up [0...99, 99...0]
            full_seq = [single_rast if i % 2 == 0 else single_rast[::-1] for i in range(self.passes)]
            return np.concatenate(full_seq)
        
        presets = {'lr_rast' : raster_repaint(bidirectional),
                    'rl_rast' : raster_repaint(bidirectional)[::-1],
                    'max_dist': find_max_dist(self.spots_expanded),
                    'rand' : np.random.permutation(self.spots_expanded),
        }
        
        if isinstance(sequence, str):
            if sequence in presets:
                return presets[sequence]
            else:
                raise ValueError(f"Unknown preset: '{sequence}'. Available: {list(presets.keys())}")  
        return sequence

    def sim(self, period, sequence, starting_phase = None, n_phases = 36, weighting = True, region = 'ctv'):
        
        '''
        Simulates spot scanning sequence in a moving target, returning the resultant dose distribution
        Args:
            t_step (float): Time taken for proton beam to move to adjacent spots, in s
            period (float): Period of breathing motion, in s
            sequence (arr, str): Sequence for proton beam spots
        '''
        
        sequence = self.set_sequence(sequence)
        
        phase = 2*np.pi/period # Phase of breathing
        
        nx = self.n_spots[0] # Number of columns in spot grid
        
        # Decode spot sequence into row and columns
        cols = sequence % nx
        rows = sequence // nx
        
        # Calculate grid distance traveled in X and Y
        jumps_x = np.abs(cols[1:] - cols[:-1])
        jumps_y = np.abs(rows[1:] - rows[:-1])
        
        # Convert distances to ticks
        ticks_x = jumps_x * self.t_step_ticks_x
        ticks_y = jumps_y * self.t_step_ticks_y
        
        # Add ticks in both dimensions to get total tick time of jump using Manhatten distance
        ticks = ticks_x + ticks_y
        
        # Accumulate the integers ticks
        cumulative_ticks = np.zeros(len(sequence), dtype=int)
        cumulative_ticks[1:] = np.cumsum(ticks)
        
        # Convert ticks to seconds for physical calculations
        times = cumulative_ticks * self.tick_time
        
        if starting_phase is None:
            # Vectorize over selected phases
            phases = np.linspace(0, 2*np.pi, n_phases, endpoint=False)[:, np.newaxis]
        else:
            # Simulate just the single specified phase
            phases = np.array([starting_phase])[:, np.newaxis]

        # Resultant positional shifts in x and y
        shifts_x = self.amp_x * np.sin((phase*times) + phases) 
        shifts_y = self.amp_y * np.sin((phase*times) + phases)

        # Find the spot edges in sequence order and apply time spatial shifts in x and y
        left_shifts = (self.spot_left[sequence] - shifts_x)[:, :, np.newaxis]
        right_shifts = (self.spot_right[sequence] - shifts_x)[:, :, np.newaxis]
        bottom_shifts = (self.spot_bottom[sequence] - shifts_y)[:, :, np.newaxis]
        top_shifts = (self.spot_top[sequence] - shifts_y)[:, :, np.newaxis]
        
        # Mask for proton spots over the env
        mask_x = (self.env_space_x > left_shifts) & (self.env_space_x < right_shifts)
        mask_y = (self.env_space_y > bottom_shifts) & (self.env_space_y < top_shifts)
        spot_masks = mask_x & mask_y

        # Add dose of 1 at every part of env space visited by a proton spot
        dose_dist = np.sum(spot_masks, axis = 1)
        
        dose_dist = dose_dist if starting_phase is None else dose_dist[0] # Array fix for a starting phase
        
        error = self.calc_error(dose_dist, weighting = weighting, region = region) # Calculate the error function of sim

        return dose_dist, error   

    def get_intended_dist(self):
        '''
        Deliver the intended distribution to env
        '''
        
        # Collect spot edges
        left_edges = self.spot_left[:, np.newaxis]
        right_edges = self.spot_right[:, np.newaxis]
        bottom_edges = self.spot_bottom[:, np.newaxis]
        top_edges = self.spot_top[:, np.newaxis]

        # Generate spot regions using masks
        mask_x = (self.env_space_x > left_edges) & (self.env_space_x < right_edges)
        mask_y = (self.env_space_y > bottom_edges) & (self.env_space_y < top_edges)
        spot_masks = mask_x & mask_y
        
        return spot_masks.T @ self.spot_weights # Apply required dose to each spot
    
    def calc_error(self, dose_dist, weighting = True, region = 'ctv'):
        '''
        Calculates the error of a resultant dose distribution against the intended distribution.
        If weighting=True, applies a penalty (x5) to underdosed voxels (WMSE).
        Otherwise, returns standard MSE.
        '''
        
        # Check which region is evaluated and mask:
        if region == 'ctv':
            region_mask = (np.abs(self.env_space_x) <= self.ctv_half_x) & (np.abs(self.env_space_y) <= self.ctv_half_y)
        elif region == 'itv':
            region_mask = (np.abs(self.env_space_x) <= self.itv_extent_x) & (np.abs(self.env_space_y) <= self.itv_extent_y)
        else:
            raise ValueError("Region must be 'ctv' or 'itv'")
        
        # Check for starting phase and apply mask
        region_dose = dose_dist[region_mask] if dose_dist.ndim == 1 else dose_dist[:, region_mask]
        
        # Collect intended distributed in required region
        intended_dist = self.get_intended_dist()
        intended_dose= intended_dist[region_mask]
        
        # Calculate raw differences
        diffs = region_dose - intended_dose
        sq_diffs = diffs**2
        
        # Apply penalty to negative differences (underdosing) if weighting is on
        if weighting:
            sq_diffs[diffs < 0] *= 5
            
        return np.mean(sq_diffs)
          
    def calculate_mask_tensor(self, period, starting_phase = None, n_phases = 36, region = 'ctv'):
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
        
        # Generate the row anmd column value of each required spot
        nx = self.n_spots[0]
        cols = self.spots_expanded % nx
        rows = self.spots_expanded // nx
        
        # Find the max path in each dimension
        max_path_x = 2 * np.sum(np.abs(cols - np.median(cols)))
        max_path_y = 2 * np.sum(np.abs(rows - np.median(rows)))
        
        # Convert to ticks
        worst_ticks_x = max_path_x * self.t_step_ticks_x
        worst_ticks_y = max_path_y * self.t_step_ticks_y
        
        # Add worst cases for maximum time
        t_max_ticks = int(worst_ticks_x + worst_ticks_y)
        
        # Convert to secs
        times = (np.arange(t_max_ticks + 1) * self.tick_time)[np.newaxis, :]
    
        # Determine phase list from start_phase argument
        if starting_phase is None:
            # Use standard number of phases
            phases = np.linspace(0, 2*np.pi, n_phases, endpoint=False)[:, np.newaxis]
        else:
            phases = np.array([starting_phase])[:, np.newaxis]
            
        # Create shift matrices
        shifts_x = self.amp_x * np.sin((phase_speed * times) + phases)
        shifts_y = self.amp_y * np.sin((phase_speed * times) + phases)

        # Define target region
        if region.lower() == 'ctv':
            self.tensor_mask = (np.abs(self.env_space_x) <= self.ctv_half_x) & \
                               (np.abs(self.env_space_y) <= self.ctv_half_y)
        elif region.lower() == 'itv':
            self.tensor_mask = (np.abs(self.env_space_x) <= self.itv_extent_x) & \
                               (np.abs(self.env_space_y) <= self.itv_extent_y)
        else:
            raise ValueError("Region must be 'ctv' or 'itv'")
        
        # Target voxel coords
        target_x = self.env_space_x[self.tensor_mask]
        target_y = self.env_space_y[self.tensor_mask]

        # Calculate the maximum possible number of voxels hit by a spot
        max_hits_x = int(np.ceil(self.spot_size[0] / self.res)) + 2
        max_hits_y = int(np.ceil(self.spot_size[1] / self.res)) + 2
        max_hits = max_hits_x * max_hits_y
        
        # Generate tensor for all phases, for all spots, for all time steps, for all the voxels hit
        self.tensor = _build_tensor_2d(shifts_x, shifts_y, self.spot_left, self.spot_right, self.spot_bottom, self.spot_top, target_x, target_y, max_hits)
    
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
        
        # Unpack spot numbers in x and y
        nx = self.n_spots[0]
        cols = sequences % nx
        rows = sequences // nx
        
        # Calculate jump distances in each axis
        jumps_x = np.abs(np.diff(cols, axis=1))
        jumps_y = np.abs(np.diff(rows, axis=1))

        # Convert distances to additive ticks
        ticks = (jumps_x * self.t_step_ticks_x) + (jumps_y * self.t_step_ticks_y)

        # Finding the cumulative jump distances as tensor indices
        time_indices = np.zeros(sequences.shape, dtype=int)
        time_indices[:, 1:] = np.cumsum(ticks, axis=1)
        
        intended_mask = self.get_intended_dist()[self.tensor_mask]
        
        return _evaluation_core(self.tensor, intended_mask, sequences, time_indices, weighting) 
    
    def display(self, doses_dict):
        '''
        Plots the intended distribution plus any number of simulated sequences.
        Args:
            doses_dict (dict): A dictionary where keys are titles (strings) 
                               and values are the dose arrays (1D flat).
        '''
        # Always include the intended distribution as the first plot
        intended_2d = self.get_intended_dist().reshape(self.n_voxels_y, self.n_voxels_x)
        
        # Prepare titles and 2D arrays
        titles = ["Intended (ITV)"] + list(doses_dict.keys())
        all_doses = [intended_2d]
        for d in doses_dict.values():
            # If d has 2 dimensions (Phases, Voxels), take the mean across phases
            if d.ndim == 2:
                d_to_plot = np.mean(d, axis=0)
            else:
                d_to_plot = d
                
            all_doses.append(d_to_plot.reshape(self.n_voxels_y, self.n_voxels_x))
        
        # Find the global max dose to standardise the color scale
        global_vmax = max([d.max() for d in all_doses])
        global_vmin = 0

        n_plots = len(all_doses)
        cols = min(n_plots, 3)
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), squeeze=False)
        extent = [-self.total_env_half_x, self.total_env_half_x, -self.total_env_half_y, self.total_env_half_y]

        for i, ax in enumerate(axes.flat):
            if i < n_plots:
                im = ax.imshow(all_doses[i], extent=extent, origin='lower', 
                               cmap='jet', vmin=global_vmin, vmax=global_vmax)
                
                ax.set_title(titles[i])
                ax.set_xlabel("X (cm)")
                ax.set_ylabel("Y (cm)")
                ax.set_aspect('equal')
                fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
            else:
                ax.axis('off')

        plt.tight_layout()
        plt.show()
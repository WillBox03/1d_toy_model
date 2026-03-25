import numpy as np
import math
import matplotlib.pyplot as plt
from numba import njit, prange

###################################################
# NUMBA CORES AND SHARED MATH HELPER FUNCTIONS
###################################################

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

@njit(cache=True)
def _calculate_hi(phase_dose, ctv_indices, max_expected_dose=5.0, n_bins=1000):
    """
    Calculates D2, D98 and D50 for the homogeneity index
    """
    # Pre-allocate histogram
    hist = np.zeros(n_bins, dtype=np.int32)
    bin_width = max_expected_dose / n_bins
    
    n_ctv_voxels = len(ctv_indices)
    
    # Tally voxels into bins
    for i in range(n_ctv_voxels):
        v = ctv_indices[i]
        d = phase_dose[v]
        
        bin_idx = int(d / bin_width)
        
        # Clamp to prevent out of bounds errors
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        elif bin_idx < 0:
            bin_idx = 0
            
        hist[bin_idx] += 1
        
    # Safety check
    if n_ctv_voxels == 0: return 0.0
        
    #  Calculate the voxel thresholds for 2%, 50% and 98% of the volume
    v_2 = int(n_ctv_voxels * 0.02)
    v_98 = int(n_ctv_voxels * 0.98)
    v_50 = int(n_ctv_voxels * 0.5)
    
    # Find D2 (Start from the hottest bin and count down)
    cumulative_voxels = 0
    d2_val = max_expected_dose
    for b in range(n_bins - 1, -1, -1):
        cumulative_voxels += hist[b]
        if cumulative_voxels >= v_2:
            d2_val = b * bin_width
            break
    
    # Find D50
    for b in range(b, -1, -1): # Continue counting down
        cumulative_voxels += hist[b]
        if cumulative_voxels >= v_50:
            d50_val = b * bin_width
            break
            
    # Find D98
    for b in range(b, -1, -1): # Continue counting down
        cumulative_voxels += hist[b]
        if cumulative_voxels >= v_98:
            d98_val = b * bin_width
            break
            
    # Return the Homogeneity Index
    return (d2_val - d98_val) / d50_val

@njit(cache = True)
def _evaluation_core_single_thread(tensor, intended, ctv_indices, seq, time_indices, weighting, w_mse, w_hi, w_pmax, w_vmax):
    
    n_phases = tensor.shape[0]
    n_voxels = len(intended)
    max_hits = tensor.shape[3]
    seq_len = seq.shape[0]
    
    total_sq_diff = 0.0
    worst_voxel_mse = 0.0
    worst_phase_mse = 0.0
    total_hi = 0.0
    
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
                
        phase_sq_diff = 0.0
                
        # Calculate Error
        for v in range(n_voxels):
            diff = phase_dose[v] - intended[v]
            sq_diff = diff * diff
            
            # Apply penalty only if it's a cold spot
            if weighting and diff < 0:
                sq_diff *= 5.0
                abs_err = abs(diff) * 5.0
            else:
                abs_err = abs(diff)
                
            phase_sq_diff += sq_diff
            
            # Track global worst voxel error
            if abs_err > worst_voxel_mse:
                worst_voxel_mse = abs_err
            
            global_max_error = max(global_max_error, abs_err)
                
        phase_mse = phase_sq_diff / n_voxels
        
        # Tracking the worst phase
        if phase_mse > worst_phase_mse:
            worst_phase_mse = phase_mse
            
        # Add to the tracker for the Average MSE
        total_sq_diff += phase_sq_diff
        
        # Calculate homogeneity index for this phase
        dynamic_max = np.max(intended) * 3.0    
        phase_hi = _calculate_hi(phase_dose, ctv_indices, max_expected_dose=dynamic_max, n_bins=1000)
        total_hi += phase_hi
        
    avg_mse = total_sq_diff / (n_phases * n_voxels) # Average MSE
    avg_hi = total_hi / n_phases
            
    return (w_mse * avg_mse) + (w_hi * avg_hi) + (w_pmax * worst_phase_mse) + (w_vmax * worst_voxel_mse)

@njit(parallel=True, cache=True)
def _evaluation_core(tensor, intended, sequences, time_indices, weighting):
    
    n_sequences = sequences.shape[0]
    n_phases = tensor.shape[0]
    max_hits = tensor.shape[3]
    n_voxels = len(intended)
    seq_len = sequences.shape[1]
    
    # Pre-allocate output arrays
    avg_mses = np.zeros(n_sequences)
    worst_mses = np.zeros(n_sequences)
    avg_dists = np.zeros((n_sequences, n_voxels))
    worst_dists = np.zeros((n_sequences, n_voxels))
    std_mses = np.zeros(n_sequences)
    
    for i in prange(n_sequences):
        seq_sum_mse = 0.0
        seq_sum_sq_mse = 0.0
        seq_worst_mse = -1.0
        
        # Temporary arrays to hold cumulative/worst doses for this specific sequence
        seq_avg_dose = np.zeros(n_voxels)
        seq_worst_dose = np.zeros(n_voxels)
        
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
            phase_sq_diff = 0.0
            for v in range(n_voxels):
                diff = phase_dose[v] - intended[v]
                sq_diff = diff * diff
                
                if weighting and diff < 0:
                    sq_diff *= 5.0
                    
                phase_sq_diff += sq_diff
                
                # Accumulate the dose for the average distribution calculation
                seq_avg_dose[v] += phase_dose[v]
                
            phase_mse = phase_sq_diff / n_voxels # Average phase MSE
            seq_sum_mse += phase_mse
            seq_sum_sq_mse += (phase_mse * phase_mse) # Square MSE, add to total for std
            
            # Check if this is the worst phase for this sequence
            if p == 0 or phase_mse > seq_worst_mse:
                seq_worst_mse = phase_mse
                # Save the worst dose distribution
                for v in range(n_voxels):
                    seq_worst_dose[v] = phase_dose[v]
                    
        # Finalize the sequence and calculate standard deviation
        avg_mse = seq_sum_mse / n_phases
        avg_mses[i] = avg_mse
        worst_mses[i] = seq_worst_mse
        var = (seq_sum_sq_mse / n_phases) - (avg_mse * avg_mse)
        std_mses[i] = math.sqrt(max(0.0, var)) # 0 divide error protection
        
        for v in range(n_voxels):
            avg_dists[i, v] = seq_avg_dose[v] / n_phases
            worst_dists[i, v] = seq_worst_dose[v]
            
    return avg_mses, worst_mses, avg_dists, worst_dists

def calculate_tick_constants(*times_in_seconds):
    """Finds the GCD tick time and tick multipliers for any number of time constants."""
    multiplier = 100000 # Correct for floating point errors
    int_times = [int(np.round(t * multiplier)) for t in times_in_seconds]
    gcd_val = math.gcd(*int_times) if len(int_times) > 1 else int_times[0] # Engine tick time
    tick_time = gcd_val / multiplier
    ticks = [int(np.round(t / tick_time)) for t in times_in_seconds]
    return tick_time, ticks

def generate_phases(n_phases, starting_phase):
    """Generates the phase column vector"""
    if starting_phase is None:
        return np.linspace(0, 2*np.pi, n_phases, endpoint=False)[:, np.newaxis]
    else:
        return np.array([starting_phase])[:, np.newaxis]
    
def generate_phase_shifts(times, phases, period, motion):
    """Generates the phase shifts at each time step for the selected motion type"""
    if motion == 'sin':
        phase_speed = 2 * np.pi / period 
        return np.sin((phase_speed * times) + phases)
    elif motion == 'sin6':
        phase_speed = np.pi / period
        return (2.0 * (np.sin((phase_speed * times) + phases) ** 6)) - 1.0
    else:
        raise ValueError(f"Unknown motion model: '{motion}'. Use 'sin' or 'sin6'.")
    
def calc_mse(dose_dist, intended_dist, region_mask, weighting=True):
        '''Calculates the MSE of a resultant dose distribution.'''
        region_dose = dose_dist[region_mask] if dose_dist.ndim == 1 else dose_dist[:, region_mask] # Check for starting phase and apply mask
        intended_dose = intended_dist[region_mask]
        
        # Get squared errors of dose
        diffs = region_dose - intended_dose
        sq_diffs = diffs**2
        
        # If using weighted MSE, multiply negative errors
        if weighting:
            sq_diffs[diffs < 0] *= 5.0
            
        return np.mean(sq_diffs)

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

###################################################
# CLASS FOR 1D environments
###################################################
class ITV_env_1D:   
    '''
    Environment space for the simulation in 1 dimension
    '''
    def __init__(self, res, ctv_length, amp, n_spots = None, spot_size = None, spacing = None, t_step = 0.1, t_switch = 0.5, margin = None):
        '''
        Initialises the environment, creating a high resolution coordinate space for the env of set length. Also defines positions of proton beam spots covering the ITV
        Args:
            res (float): Resolution of environment, in cm
            ctv_length (float,int): Length of env, in cm
            n_spots (int): Number of proton beam spots covering the ITV
            amp(float): Standard amplitude (half of the peak-to-peak amplitude) of breathing motion, in cm
            spot_size (float): Physical width of spot in cm
        '''
            
        if margin is None:
            margin = amp
        
        self.update_env(res=res, ctv_length=ctv_length, amp=amp, n_spots=n_spots, spot_size=spot_size, spacing=spacing, t_step=t_step, t_switch=t_switch, margin=margin)
        
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
        if 'spacing' in kwargs: self.spacing = kwargs['spacing']
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
        
        spot_keys = ['n_spots', 'spot_size', 'spacing']
        spot_args = {k: kwargs[k] for k in spot_keys if k in kwargs and kwargs[k] is not None}
        
        if spot_args:
            # Spot params passed in, must be exactly 2
            if len(spot_args) != 2:
                raise ValueError("When updating the grid, you must provide exactly two of: n_spots, spot_size, spacing.")
            n_in = spot_args.get('n_spots')
            size_in = spot_args.get('spot_size')
            sp_in = spot_args.get('spacing')
        else:
            # No spot params passed in, default to recalculating spacing
            n_in = getattr(self, 'n_spots', None)
            size_in = getattr(self, 'spot_size', None)
            sp_in = None
            
        if n_in is not None and size_in is not None and sp_in is None:
            # Calculate spacing
            sp_in = (total_itv_width - size_in) / (n_in - 1) if n_in > 1 else 0.0
            
        elif size_in is not None and sp_in is not None and n_in is None:
            # Calculate number of spots
            if sp_in > 0:
                n_in = int(np.ceil((total_itv_width - size_in) / sp_in)) + 1 
                sp_in = (total_itv_width - size_in) / (n_in - 1) if n_in > 1 else 0.0 # Recalculate spacing if needed to snap spots to ITV
            else:
                n_in = 1
        elif n_in is not None and sp_in is not None and size_in is None:
            # Calculate spot size
            size_in = total_itv_width - (n_in - 1) * sp_in
            if size_in <= 0:
                raise ValueError("Spacing too large for the given number of spots in this ITV.")
        else:
            raise ValueError("You must provide exactly two of: 'n_spots', 'spot_size', or 'spacing'.")

        # Save spot geometry
        self.n_spots = n_in
        self.spot_size = size_in
        self.spacing = sp_in
        
        print(f"Spot Grid: {n_in} spots | Size: {size_in:.2f}cm | Spacing: {sp_in:.2f}cm")
        
        # Define spot centres and edges
        spot_centres = -self.itv_extent + self.spot_size/2 + (np.arange(self.n_spots) * self.spacing)
        
        # Define proton spot edges in environment
        self.spot_left_edges = spot_centres - self.spot_size / 2
        self.spot_right_edges = spot_centres + self.spot_size / 2
        
        # Set tick time and step ticks
        self.tick_time, (self.t_step_ticks, self.t_switch_ticks) = calculate_tick_constants(self.t_step, self.t_switch)

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

    def sim(self, period, sequence, starting_phase = None, n_phases = 24, weighting = True, mode = '1d', region = 'itv', motion = 'sin'):
        
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
        
        jumps = abs(sequence[1:] - sequence[:-1]) # Distance between each proton beam spot in sequence

        ticks = jumps * self.t_step_ticks
        
        # Apply mode penalty to 0 jumps
        ticks[jumps == 0] = self.mode_presets.get(mode, 0)
        
        # Accumulate the integers ticks
        cumulative_ticks = np.zeros(len(jumps) + 1, dtype=int)
        cumulative_ticks[1:] = np.cumsum(ticks)
        
        # Convert ticks to seconds for physical calculations
        times = cumulative_ticks * self.tick_time
        
        phases = generate_phases(n_phases, starting_phase)

        shifts = generate_phase_shifts(times, phases, period, motion)

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

    def get_nominal_dist(self):
        '''
        Deliver the intended distribution to env
        '''
        left_edges = self.spot_left_edges[:, np.newaxis]
        right_edges = self.spot_right_edges[:, np.newaxis]
    
        spot_masks = (self.env_space > left_edges) & (self.env_space < right_edges)
        planned_dose = (spot_masks.T @ self.spot_weights).astype(np.float64)
        
        intended = np.zeros(self.n_env_voxels, dtype=np.float64)
        ctv_mask = self.get_region_mask('ctv')
        intended[ctv_mask] = planned_dose[ctv_mask]
        
        return intended
    
    def get_region_mask(self, region):
        '''Check which region is evaluated and returns mask of environment'''
        if region == 'ctv':
            return (self.env_space >= -self.ctv_half_length) & (self.env_space <= self.ctv_half_length)
        elif region == 'itv':
            return (self.env_space >= -self.itv_extent) & (self.env_space <= self.itv_extent)
        else:
            raise ValueError("Region must be 'ctv' or 'itv'")
        
    def calc_error(self, dose_dist, weighting=True, region='itv'):
        """Calculate the error function of the simulated dose distribution from the intended distribution"""
        mask = self.get_region_mask(region)
        intended = self.get_nominal_dist()
        return calc_mse(dose_dist, intended, mask, weighting)
          
    def calculate_mask_tensor(self, period, starting_phase = None, n_phases = 24, mode = '1d', region = 'itv', motion = 'sin'):
        """
        Precomputes all possible spot/time combinations.
    
        Args:
            timestep (float): Time per jump unit.
            period (float): Breathing period.
            starting_phase (float/None): 
                If float: returns 3D Tensor (Spots, Time, Voxels)
                If None: returns 4D Tensor (All Phases, Spots, Time, Voxels)
        """
        
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
    
        phases = generate_phases(n_phases, starting_phase)
            
        shifts = generate_phase_shifts(times, phases, period, motion) # Create shift matrix
        
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
        
        intended_mask = self.get_nominal_dist()[self.tensor_mask]
        
        return _evaluation_core(self.tensor, intended_mask, sequences, time_indices, weighting)
    
    def get_dvh(self, dose_dist, region='itv'):
        """Calculates the Dose Volume Histogram (DVH) of a given dose distribution"""
        mask = self.get_region_mask(region)
        roi_doses = dose_dist[mask].flatten() # Get dose from required region
            
        sorted_doses = np.sort(roi_doses) # Sort doses
        n_voxels = len(sorted_doses)
        vol_pct = np.linspace(100, 0, n_voxels) # Create a volume axis
        
        # Prepend a 0-dose point
        if sorted_doses[0] > 0:
            sorted_doses = np.insert(sorted_doses, 0, 0.0)
            vol_pct = np.insert(vol_pct, 0, 100.0)
        
        return sorted_doses, vol_pct
    
    def display(self, sims = None, labels = None, avg_phases = False, spot_edges = False):
        '''
        Generate and display the intended target distribution, along with other simulated distributions
        '''
        
        target_dist = self.get_nominal_dist()
        
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
        
    def display_dvh(self, dose_dict, regions=None):
        """Plots a clinical DVH"""      
        if regions is None:
            regions = ['ctv', 'itv']
            
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10.colors
        line_styles = {'ctv': '-', 'itv': '--', 'body': ':'}
        
        for idx, (name, dose_dist) in enumerate(dose_dict.items()):
            # Average the phases if a multi-phase array is passed
            if dose_dist.ndim > 1 and dose_dist.shape[0] < 100: 
                plot_dose = np.mean(dose_dist, axis=0) 
            else:
                plot_dose = dose_dist
                
            color = colors[idx % len(colors)]
            
            for region in regions:
                try:
                    x_dose, y_vol = self.get_dvh(plot_dose, region=region)
                    style = line_styles.get(region, '-')
                    label = f"{name} ({region.upper()})"
                    plt.plot(x_dose, y_vol, linestyle=style, color=color, linewidth=2, label=label)
                except Exception as e:
                    print(f"Skipping {region} for {name}: {e}")

        # Add target line
        rx_dose = np.max(self.get_nominal_dist())
        plt.axvline(x=rx_dose, color='black', linestyle='-.', alpha=0.5, label='Prescription Dose')

        plt.title("Cumulative Dose Volume Histogram (DVH) - 1D")
        plt.xlabel("Dose (Arbitrary Units)")
        plt.ylabel("Volume (%)")
        plt.ylim(0, 105)
        plt.xlim(left=0)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()
        
        
###################################################
# CLASS FOR 2D environments
###################################################
class ITV_env_2D:
    '''
    Environment space for the simulation in 2D
    '''
    def __init__(self, res, ctv_size, amp, n_spots = None, spot_size = None, spacing = None, t_steps = (0.1,0.1), margin = None):
        '''
        Initialises the environment, creating a high resolution coordinate space for the env of set length. Also defines positions of proton beam spots covering the ITV
        Args:
            res (float): Resolution of environment, in cm
            ctv_size (tuple, float): Dimensions of CTV in x and y, in cm
            n_spots (tuple, int): Number of proton beam spots covering the ITV in x and y dims
            amp(float): Standard amplitude (half of the peak-to-peak amplitude) of breathing motion, in both x and y dims, in cm
            spot_size (float, float/None): Physical width of spot in cm
            t_steps (tuple, float): Time taken for proton beam to move to adjacent spots, in x and y dims, in s
            margin(tuple, float): Space between edge of ITV and edge of simulation in x and y dims
        '''
            
        if margin is None:
            margin = amp
        
        self.update_env(res=res, ctv_size=ctv_size, amp=amp, n_spots=n_spots, spot_size=spot_size, spacing=spacing, t_steps=t_steps, margin=margin)
                
    def update_env(self, **kwargs):
        '''
        Update the number of proton spots and the oscillation amplitude without initialising a new object
        '''
        
        if 'res' in kwargs: self.res = kwargs['res']
        if 'ctv_size' in kwargs: self.ctv_size = kwargs['ctv_size']
        if 'amp' in kwargs: self.amp = kwargs['amp']
        if 't_steps' in kwargs: self.t_steps = kwargs['t_steps']
        if 'margin' in kwargs: self.margin = kwargs['margin']
        
        spot_keys = ['n_spots', 'spot_size', 'spacing']
        spot_args = {k: kwargs[k] for k in spot_keys if k in kwargs and kwargs[k] is not None}
        
        if spot_args:
            # Spot params passed in, must be 2
            if len(spot_args) != 2:
                raise ValueError("When updating the grid, you must provide exactly two of: n_spots, spot_size, spacing.")
            n_in= spot_args.get('n_spots')
            size_in = spot_args.get('spot_size')
            sp_in = spot_args.get('spacing')
        else:
            # No spot params passed in with updated env, sdefault to recalculate spacing
            n_in = getattr(self, 'n_spots', None)
            size_in = getattr(self, 'spot_size', None)
            sp_in = None
        
        def spot_solve_axis(total_width, n, size, sp):
            ''' Helper to solve the spot geomerty for an axis'''
            if n is not None and size is not None and sp is None:
                # Calculate spacing
                if n > 1:
                    sp = (total_width - size) / (n - 1)
                else:
                    sp = 0.0
            elif size is not None and sp is not None and n is None:
                # Calculate number of spots
                if sp > 0:
                    n = int(np.ceil((total_width - size) / sp)) + 1
                    sp = (total_width - size) / (n - 1) if n > 1 else 0.0 # Adjust spacing to snap spots to ITV
                else:
                    n = 1
            elif n is not None and sp is not None and size is None:
                # Calculate spot size
                size = total_width - (n - 1) * sp
                if size <= 0:
                    raise ValueError("Spacing is too large for the given number of spots.")
            else:
                raise ValueError("You must provide exactly two of: n_spots, spot_size, or spacing (as tuples).")
            return n, size, sp
        
        # Unpack CTV size, amplitude and margins
        self.ctv_half_x, self.ctv_half_y = self.ctv_size[0] / 2, self.ctv_size[1] / 2
        self.amp_x, self.amp_y = self.amp
        self.margin_x, self.margin_y = self.margin
        
        # Boundaries of ITV, and lengths of the environment
        self.itv_extent_x = self.ctv_half_x + self.amp_x
        self.itv_extent_y = self.ctv_half_y + self.amp_y
        
        self.total_env_half_x = self.itv_extent_x + self.margin_x
        self.total_env_half_y = self.itv_extent_y + self.margin_y
        
        # Total widths of itv
        total_width_x = 2 * self.itv_extent_x
        total_width_y = 2 * self.itv_extent_y
        
        # Resolve spot geometry and save to env
        nx, sx, spx = spot_solve_axis(total_width_x, n_in[0] if n_in else None, size_in[0] if size_in else None, sp_in[0] if sp_in else None)
        ny, sy, spy = spot_solve_axis(total_width_y, n_in[1] if n_in else None, size_in[1] if size_in else None, sp_in[1] if sp_in else None)
        self.n_spots = (nx, ny)
        self.spot_size = (sx, sy)
        self.spacing = (spx, spy)
        self.n_spots_tot = nx * ny
        
        print(f"Spot Grid: {nx}x{ny} spots | Size: {sx:.2f}x{sy:.2f}cm | Spacing: {spx:.2f}x{spy:.2f}cm")
        
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
        spot_axis_x = -self.itv_extent_x + sx/2 + (np.arange(nx) * spx)
        spot_axis_y = -self.itv_extent_y + sy/2 + (np.arange(ny) * spy)
        SX, SY = np.meshgrid(spot_axis_x, spot_axis_y)
        self.spot_centres_x = SX.flatten()
        self.spot_centres_y = SY.flatten()
        
        # Define proton spot edges in environment
        self.spot_left = self.spot_centres_x - spot_size_x / 2
        self.spot_right = self.spot_centres_x + spot_size_x / 2
        self.spot_bottom = self.spot_centres_y - spot_size_y / 2
        self.spot_top = self.spot_centres_y + spot_size_y / 2
        
        # Set tick time and step ticks
        self.tick_time, (self.t_step_ticks_x, self.t_step_ticks_y) = calculate_tick_constants(self.t_steps[0], self.t_steps[1])

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

    def sim(self, period, sequence, starting_phase = None, n_phases = 24, weighting = True, region = 'itv', motion = 'sin'):
        
        '''
        Simulates spot scanning sequence in a moving target, returning the resultant dose distribution
        Args:
            t_step (float): Time taken for proton beam to move to adjacent spots, in s
            period (float): Period of breathing motion, in s
            sequence (arr, str): Sequence for proton beam spots
        '''
        
        sequence = self.set_sequence(sequence)
        
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
        
        phases = generate_phases(n_phases, starting_phase)
            
        shifts = generate_phase_shifts(times, phases, period, motion)   

        # Resultant positional shifts in x and y
        shifts_x = self.amp_x * shifts
        shifts_y = self.amp_y * shifts
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
        
        error_metrics = self.calc_error_metrics(dose_dist, weighting = weighting, region = region) # Calculate the error function of sim

        return dose_dist, error_metrics 

    def get_nominal_dist(self):
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
        
        planned_dose = (spot_masks.T @ self.spot_weights).astype(np.float64)
        
        intended = np.zeros(self.n_env_voxels, dtype=np.float64)
        ctv_mask = self.get_region_mask('itv')
        
        intended[ctv_mask] = planned_dose[ctv_mask]
        
        return intended
    
    def get_4d_composite(self, n_phases = 24, motion='sin'):
        '''
        Calculates the 4D-Composite dose for also phase distributions free of interplay.
        Represents a perfectly motion-blurred distribution
        '''
        # Time is 0 for all spots
        times = np.array([[0.0]]) 
        phases = generate_phases(n_phases, None)
        # Generating shifts with only starting phases as arguments
        shifts = generate_phase_shifts(times, phases, 1.0, motion) 
        
        shifts_x = self.amp_x * shifts
        shifts_y = self.amp_y * shifts
        
        composite_dose = np.zeros(self.n_env_voxels, dtype=np.float64)
        
        for p in range(n_phases):
            # Shift the spot edges for this specific phase
            left_edges = (self.spot_left - shifts_x[p, 0])[:, np.newaxis]
            right_edges = (self.spot_right - shifts_x[p, 0])[:, np.newaxis]
            bottom_edges = (self.spot_bottom - shifts_y[p, 0])[:, np.newaxis]
            top_edges = (self.spot_top - shifts_y[p, 0])[:, np.newaxis]

            # Generate spot regions using masks for this phase
            mask_x = (self.env_space_x > left_edges) & (self.env_space_x < right_edges)
            mask_y = (self.env_space_y > bottom_edges) & (self.env_space_y < top_edges)
            spot_masks = mask_x & mask_y
            
            # Calculate what the dose would be for this phase
            phase_dose = (spot_masks.T @ self.spot_weights).astype(np.float64)
            composite_dose += phase_dose
            
        # The 4D Composite is the average of all phases
        return composite_dose / n_phases
        
    
    def get_region_mask(self, region):
        '''Check which region is evaluated and returns mask of environment'''
        if region == 'ctv': 
            return (np.abs(self.env_space_x) <= self.ctv_half_x) & (np.abs(self.env_space_y) <= self.ctv_half_y)
        elif region == 'itv': 
            return (np.abs(self.env_space_x) <= self.itv_extent_x) & (np.abs(self.env_space_y) <= self.itv_extent_y)
        else:
            raise ValueError("Region must be 'ctv' or 'itv'")
        
    def calc_error_metrics(self, dose_dist, weighting=True, region='itv', motion='sin'):
        """Calculate MSE, maximum error, Homogeneity Index, and WOrst Phase MSE"""
        
        if dose_dist.ndim == 1:
            dose_dist = dose_dist[np.newaxis, :]
            
        n_phases = dose_dist.shape[0]
        
        # 2. Get Targets and Masks
        target_dose = self.get_4d_composite(motion=motion)
        mask_eval = self.get_region_mask(region) # Usually the ITV
        mask_ctv = self.get_region_mask('ctv')   # Strict CTV
        
        target_eval = target_dose[mask_eval]
        
        # Initialise metrics
        total_mse = 0.0
        worst_phase_mse = 0.0
        global_max_err = 0.0
        total_hi = 0.0
        
        # Evaluate by phase
        for p in range(n_phases):
            # Slice out the specific regions for this phase
            phase_eval = dose_dist[p][mask_eval]
            phase_ctv = dose_dist[p][mask_ctv]
            
            # MSE and worst-voxels errors
            diff = phase_eval - target_eval
            # Apply weighting the MSE
            weights = np.ones_like(diff)
            if weighting:
                weights[diff < 0] = 5.0
                
            # Track Phase MSE
            sq_diff = (diff ** 2) * weights
            phase_mse = np.mean(sq_diff)
            total_mse += phase_mse
            
            # Track the worst phase
            if phase_mse > worst_phase_mse:
                worst_phase_mse = phase_mse
            
            # Track maximum voxel error
            phase_max = np.max(np.abs(diff) * weights)
            if phase_max > global_max_err:
                global_max_err = phase_max
                
            # Homogeneity Index
            # D2: Hottest 2% of volume
            # D98: Coldest 2% of volume
            # D50: Average
            if len(phase_ctv) > 0:
                d2 = np.percentile(phase_ctv, 98)
                d50 = np.percentile(phase_ctv, 50)
                d98 = np.percentile(phase_ctv, 2)
                
                # HI calculation
                if d50 > 0.001: # Safety check
                    phase_hi = (d2 - d98) / d50
                else:
                    phase_hi = 1.0 # Heavy penalty if the tumor gets no dose
            else:
                phase_hi = 1.0 
                
            total_hi += phase_hi

        # Average the metrics
        avg_mse = total_mse / n_phases
        avg_hi = total_hi / n_phases
        
        # Build scorecard
        metrics = {
            'MSE (Average)': avg_mse,
            'HI (Average)': avg_hi,
            'Worst Voxel Error': global_max_err
        }
        
        # Add Multi-Phase specific metrics
        if n_phases > 1:
            metrics['Worst Phase MSE'] = worst_phase_mse
            
        return metrics
          
    def calculate_mask_tensor(self, period, starting_phase = None, n_phases = 24, region = 'itv', motion = 'sin'):
        """
        Precomputes all possible spot/time combinations.
    
        Args:
            timestep (float): Time per jump unit.
            period (float): Breathing period.
            starting_phase (float/None): 
                If float: returns 3D Tensor (Spots, Time, Voxels)
                If None: returns 4D Tensor (All Phases, Spots, Time, Voxels)
        """
        
        self.tensor_motion = motion
        
        # Generate the row and column value of each required spot
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
    
        phases = generate_phases(n_phases, starting_phase)
            
        shifts = generate_phase_shifts(times, phases, period, motion)
            
        # Create shift matrices
        shifts_x = self.amp_x * shifts
        shifts_y = self.amp_y * shifts

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
        
        intended_mask = self.get_4d_composite(motion=self.tensor_motion)[self.tensor_mask]
        
        return _evaluation_core(self.tensor, intended_mask, sequences, time_indices, weighting)
    
    def get_dvh(self, dose_dist, region='itv'):
        """Calculates the Dose Volume Histogram (DVH) of a given dose distribution"""
        mask = self.get_region_mask(region)
        roi_doses = dose_dist[mask].flatten() # Get dose from required region
            
        sorted_doses = np.sort(roi_doses) # Sort doses
        n_voxels = len(sorted_doses)
        vol_pct = np.linspace(100, 0, n_voxels) # Create a volume axis
        
        # Prepend a 0-dose point
        if sorted_doses[0] > 0:
            sorted_doses = np.insert(sorted_doses, 0, 0.0)
            vol_pct = np.insert(vol_pct, 0, 100.0)
        
        return sorted_doses, vol_pct
    
    def display(self, doses_dict):
        '''
        Plots the intended distribution plus any number of simulated sequences.
        Args:
            doses_dict (dict): A dictionary where keys are titles (strings) 
                               and values are the dose arrays (1D flat).
        '''
        # Include Nominal and Composite distributions
        intended_2d = self.get_nominal_dist().reshape(self.n_voxels_y, self.n_voxels_x)
        composite = self.get_4d_composite(motion=self.tensor_motion).reshape(self.n_voxels_y, self.n_voxels_x)
        
        # Prepare titles and 2D arrays
        titles = ["Nominal Distribution", "4D Composite Distribution"] + list(doses_dict.keys())
        all_doses = [intended_2d, composite]
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
        
    def display_dvh(self, dose_dict, regions=None):
        """Plots a clinical DVH"""      
        if regions is None:
            regions = ['ctv', 'itv']
            
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10.colors
        line_styles = {'ctv': '-', 'itv': '--', 'body': ':'}
        
        for idx, (name, dose_dist) in enumerate(dose_dict.items()):
            # Average the phases if a multi-phase array is passed
            if dose_dist.ndim > 1 and dose_dist.shape[0] < 100: 
                plot_dose = np.mean(dose_dist, axis=0) 
            else:
                plot_dose = dose_dist
                
            color = colors[idx % len(colors)]
            
            for region in regions:
                try:
                    x_dose, y_vol = self.get_dvh(plot_dose, region=region)
                    style = line_styles.get(region, '-')
                    label = f"{name} ({region.upper()})"
                    plt.plot(x_dose, y_vol, linestyle=style, color=color, linewidth=2, label=label)
                except Exception as e:
                    print(f"Skipping {region} for {name}: {e}")
                    
        # Add target line
        rx_dose = np.max(self.get_nominal_dist())
        plt.axvline(x=rx_dose, color='black', linestyle='-.', alpha=0.5, label='Prescription Dose')

        plt.title("Cumulative Dose Volume Histogram (DVH) - 1D")
        plt.xlabel("Dose (Arbitrary Units)")
        plt.ylabel("Volume (%)")
        plt.ylim(0, 105)
        plt.xlim(left=0)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()
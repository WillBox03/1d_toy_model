import numpy as np
import matplotlib.pyplot as plt

class ITV_env:   
    '''
    Environment space for the simulation
    '''
    def __init__(self, res, env_length, n_spots, amp, spot_size = None):
        '''
        Initialises the environment, creating a high resolution coordinate space for the env of set length. Also defines positions of proton beam spots covering the ITV
        Args:
            res (float): Resolution of environment, in cm
            env_length (float,int): Length of env, in cm
            n_spots (int): Number of proton beam spots covering the ITV
            amp(float): Amplitude of breathing motion, in cm
            spot_size (float): Physical width of spot in cm
        '''
        if spot_size is None:
            spot_size = (env_length + (2*amp)) / n_spots
        
        
        self.update_env(res=res, env_length=env_length, amp=amp, n_spots=n_spots, spot_size=spot_size)
                
    def update_env(self, **kwargs):
        '''
        Update the number of proton spots and the oscillation amplitude without initialising a new object
        '''
        
        if 'res' in kwargs: self.res = kwargs['res']
        if 'env_length' in kwargs: self.env_length = kwargs['env_length']
        if 'amp' in kwargs: self.amp = kwargs['amp']
        if 'n_spots' in kwargs: self.n_spots = kwargs['n_spots']
        if 'spot_size' in kwargs: self.spot_size = kwargs['spot_size']
        
        # Boundaries of env and ITV
        self.env_half_length = self.env_length / 2
        self.itv_extent = self.env_half_length + self.amp
        
        self.n_env_voxels = int(np.round(self.env_length / self.res)) # Calculate number of voxels used in env
        
        # Create hi-res env environment at voxel centres
        self.env_space = np.linspace(-self.env_half_length + self.res/2, self.env_half_length - self.res/2, self.n_env_voxels)
        
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
            
            if self.n_spots % 2 == 0:
                sequence =  np.roll(np.ravel((np.arange(self.n_spots//2),np.arange(self.n_spots-1, self.n_spots//2 - 1, -1)), order="F"),1)
            else:
                sequence = np.concatenate((np.array([self.n_spots//2]),np.ravel((np.arange(self.n_spots//2),np.arange(self.n_spots-1, self.n_spots//2, -1)), order="F")))
                
            return np.tile(sequence, self.passes)
            
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

    def sim(self, timestep, period, sequence, starting_phase = None, weighting = True):
        
        '''
        Simulates spot scanning sequence in a moving target, returning the resultant dose distribution
        Args:
            timestep (float): Time taken for proton beam to move to adjacent spots, in s
            period (float): Period of breathing motion, in s
            sequence (arr, str): Sequence for proton beam spots
        '''
        sequence = self.set_sequence(sequence)
        
        phase = 2*np.pi/period # Phase of breathing
        
        jumps = abs(sequence[1:] - sequence[:-1]) # Distance between each proton beam spot in sequence
    
        # Cumulative time taken to move to each proton beam spot in sequence
        times = np.zeros(len(jumps) + 1)
        
        times[1:] = np.cumsum(jumps) * timestep
        
        if starting_phase is None:
            # Vectorize over 8 phases
            phases = np.linspace(0, 2*np.pi, 8, endpoint=False)[:, np.newaxis]
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
        
        error = self.calc_error(dose_dist, weighting = weighting)

        return dose_dist, error   

    def get_intended_dist(self):
        '''
        Deliver the intended distribution to env
        '''
        
        left_edges = self.spot_left_edges[:, np.newaxis]
        right_edges = self.spot_right_edges[:, np.newaxis]
    
        spot_masks = (self.env_space > left_edges) & (self.env_space < right_edges)
        
        return spot_masks.T @ self.spot_weights
    
    def calc_error(self, dose_dist, weighting = True):
        '''
        Calculates the error of a resultant dose distribution against the intended distribution.
        If weighting=True, applies a penalty (x5) to underdosed voxels (WMSE).
        Otherwise, returns standard MSE.
        '''
        intended_dist = self.get_intended_dist()
        
        # Calculate raw differences
        diffs = dose_dist - intended_dist
        sq_diffs = diffs**2
        
        # Apply penalty to negative differences (underdosing) if weighting is on
        if weighting:
            sq_diffs[diffs < 0] *= 5
            
        return np.mean(sq_diffs)
          
    def calculate_mask_tensor(self, t_step, period, starting_phase = None):
        """
        Precomputes all possible spot/time combinations.
    
        Args:
            timestep (float): Time per jump unit.
            period (float): Breathing period.
            starting_phase (float/None): 
                If float: returns 3D Tensor (Spots, Time, Voxels)
                If None: returns 4D Tensor (All Phases, Spots, Time, Voxels)
        """
        
        self.t_step = t_step
        
        phase_speed = 2*np.pi/period # Phase of breathing
        
        # Calculate the latest possible time the simulation can run for
        # 
        spot_visits = np.sum(self.spot_weights)
        t_max = t_max = max(0, (spot_visits - 1) * (self.n_spots - 1))
        
        #Create list of times
        times = (np.arange(t_max + 1) * self.t_step)[np.newaxis, :]
    
        #Determine phase list from start_phase argument
        if starting_phase is None:
            #Use standard number of phases; here, 8
            phases = np.linspace(0, 2*np.pi, 8, endpoint=False)[:, np.newaxis]
        else:
            phases = np.array([starting_phase])[:, np.newaxis]
            
        #Create 4D tensor, currently filled with the phase shifts at each timepoint with each starting phase
        #Dimensions (P,1,T,1)
        all_shifts = (self.amp*np.sin((phase_speed*times) + phases))[:, np.newaxis, :, np.newaxis]

        #Find the left and right edges in dimensions (1, S, 1, 1)
        left_edges = self.spot_left_edges[np.newaxis, :, np.newaxis, np.newaxis]
        right_edges = self.spot_right_edges[np.newaxis, :, np.newaxis, np.newaxis]
        
        #Reshape env space to (1,1,1,V)
        env_space = self.env_space[np.newaxis, np.newaxis, np.newaxis, :]

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
        
        n_sequences = sequences.shape[0]
        intended = self.get_intended_dist()
        
        # Calculate physical jump distances
        jumps = np.abs(np.diff(sequences, axis=1))

        # Finding the cumulative jump distances aas tensor indices
        time_indices = np.zeros(sequences.shape, dtype=int)
        time_indices[:, 1:] = np.cumsum(jumps, axis=1)
        
        #Generate empty array for all mses
        evals = np.zeros(n_sequences)
        
        #Loop through each sequence
        for i in range(n_sequences):
            
            # Find the dose distributions for each beam point at the correct time for all required phases    
            dists = self.tensor[:, sequences[i], time_indices[i], :].sum(axis=1)
            
            # Find the squared differences between the distributions and intended across all phases
            # Evaluates using WMSE
            diffs = dists - intended
            
            sq_diffs = diffs**2
            
            if weighting:
                sq_diffs[diffs<0] *= 5
            
            # Average across voxels and required phases
            evals[i] = np.mean(sq_diffs)
        
        return evals
    
    def display_sims(self, sims, labels = None):
        '''
        Generate and display the intended target distribution, along with other simulated distributions
        '''
        
        target_dist = self.get_intended_dist()
                
        for i, sim in enumerate(sims):
            #Assign label if it exists
            if labels is not None and i < len(labels):
                sim_label = labels[i]
            else:
                sim_label = f'Simulated Distribution {i+1}'
                
            plt.plot(self.env_space, sim, lw=2, alpha= 0.9, label = sim_label)
            
        plt.plot(self.env_space, target_dist, 'k--', lw=2, label = "Intended Dose Distribution")
            
        plt.ylabel('Relative Dose (Times Visited)')
        plt.xlabel('Position on env (cm)')
        plt.grid(True)
        
        plt.legend()
        plt.show()

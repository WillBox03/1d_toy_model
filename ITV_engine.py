import numpy as np
import matplotlib.pyplot as plt

class ITV_env:   
    '''
    Environment space for the simulation
    '''
    def __init__(self, res, ctv_length, n_spots, amp):
        
        '''
        Initialises the environment, creating a high resolution coordinate space for the CTV of set length. Also defines positions of proton beam spots covering the ITV
        Args:
            res (float): Resolution of environment, in cm
            ctv_length (float,int): Length of CTV, in cm
            n_spots (int): Number of proton beam spots covering the ITV
            amp(float): Amplitude of breathing motion, in cm
        '''
        
        self.res = res
        self.n_spots = n_spots
        self.amp = amp # Include breathing amplitude to calculate extent of ITV
        
        # Boundaries of CTV and ITV
        self.ctv_half_length = ctv_length / 2
        self.itv_extent = self.ctv_half_length + self.amp
        
        self.n_ctv_voxels = int(np.round(ctv_length / self.res)) # Calculate number of voxels used in GTV
        
        # Create hi-res CTV environment at voxel centres
        self.ctv_space = np.linspace(-self.ctv_half_length + self.res/2, self.ctv_half_length - self.res/2, self.n_ctv_voxels)

        # Define proton spot edges in environment
        self.spot_edges = np.linspace(-self.itv_extent, self.itv_extent, self.n_spots + 1)
        
        # Define empty attributes for spot weights and dose distribution
        self.spot_weights = np.zeros(self.n_spots)
        self.dose_dist = np.zeros(self.n_ctv_voxels)
        
    def update_env(self, ctv_length, n_spots, amp):
        '''
        Update the number of proton spots and the oscillation amplitude without initialising a new object
        '''
        
        self.n_spots = n_spots
        self.amp = amp
        
        self.ctv_half_length = ctv_length / 2
        self.itv_extent = self.ctv_half_length + self.amp
        
        self.n_ctv_voxels = int(np.round(ctv_length / self.res))
        
        self.ctv_space = np.linspace(-self.ctv_half_length + self.res/2, self.ctv_half_length - self.res/2, self.n_ctv_voxels)
        
        self.spot_edges = np.linspace(-self.itv_extent, self.itv_extent, self.n_spots + 1)

        self.spot_weights = np.zeros(self.n_spots)
        self.dose_dist = np.zeros(self.n_ctv_voxels)

    def set_spot_weights(self, spot_weights):
        
        '''
        Defines the dosage weighting for each spot
        '''
        
        self.spot_weights = spot_weights


    def set_sequence(self, sequence):
        
        '''
        Defines the ordering sequence of the spots, with some preset order selections
        '''
        
        def find_max_dist():
            
            if self.n_spots % 2 == 0:
                return np.roll(np.ravel((np.arange(self.n_spots//2),np.arange(self.n_spots-1, self.n_spots//2 - 1, -1)), order="F"),1)
            else:
                return  np.concatenate((np.array([self.n_spots//2]),np.ravel((np.arange(self.n_spots//2),np.arange(self.n_spots-1, self.n_spots//2, -1)), order="F")))
            
        _presets = {'lr_rast' : np.arange(self.n_spots),
                    'rl_rast' : np.arange(self.n_spots - 1, -1, -1),
                    'rand' : np.random.permutation(self.n_spots),
                    'max_dist': find_max_dist()}
        
        if isinstance(sequence, str):
            if sequence in _presets:
                return _presets[sequence]
            else:
                raise ValueError(f"Unknown preset: '{sequence}'. Available: {list(_presets.keys())}")  
    
        return sequence

    def sim(self, timestep, period, sequence, starting_phase):
        
        '''
        Simulates spot scanning sequence in a moving target, returning the resultant dose distribution
        Args:
            timestep (float): Time taken for proton beam to move to adjacent spots, in s
            period (float): Period of breathing motion, in s
            sequence (arr, str): Sequence for proton beam spots
        '''
        self.sequence = self.set_sequence(sequence)
        
        _phase = 2*np.pi/period # Phase of breathing
        
        _jumps = abs(self.sequence[1:] - self.sequence[:-1]) # Distance between each proton beam spot in sequence
    
        # Cumulative time taken to move to each proton beam spot in sequence
        _times = np.zeros(len(_jumps) + 1)
        
        _times[1:] = np.cumsum(_jumps) * timestep

        _shifts = self.amp*np.sin((_phase*_times) + starting_phase) # Resultant positional shift at each time

        # Find the spot edges in sequence order
        _left_edges = self.spot_edges[self.sequence]
        _right_edges = self.spot_edges[self.sequence + 1]

        # Find spot edges relative to CTV position due to shifts  
        _left_shifts = (_left_edges - _shifts)[:, np.newaxis]
        _right_shifts = (_right_edges - _shifts)[:, np.newaxis]
        
        # Mask for proton spots over the CTV
        _spot_masks = (self.ctv_space > _left_shifts) & (self.ctv_space < _right_shifts)

        # Matrix multiply spot masks with dose weights to fill dose distribution
        self.dose_dist = _spot_masks.T @ self.spot_weights

        return self.dose_dist
        

    def get_intended_dist(self):
        
        '''
        Deliver the intended distribution to CTV
        '''
        
        _left_edges = self.spot_edges[:-1, np.newaxis]
        _right_edges = self.spot_edges[1:, np.newaxis]
        
        _spot_masks = (self.ctv_space > _left_edges) & (self.ctv_space < _right_edges)
        
        _intended_dist = _spot_masks.T @ self.spot_weights
        
        return _intended_dist
    
    def sim_phases(self, timestep, freq, sequence):
        
        _phases = np.linspace(0, 2*np.pi, 8, endpoint=False)

        self.dose_dists = np.array([self.sim(timestep, freq, sequence, phase) for phase in _phases])
        
        self.mse = self.calc_mse(self.dose_dists)

        return self.dose_dists, self.mse
    
    def calc_mse(self, dose_dist):
        
        '''
        Calculates the mean squared error (mse) of a resultant dose distribution against the intended distribution
        '''
        
        _intended_dist = self.get_intended_dist()
        
        _sq_diff = (dose_dist - _intended_dist)**2
        
        self.mse = np.mean(_sq_diff)
            
        return self.mse
    
    
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
        
        _phase_speed = 2*np.pi/period # Phase of breathing
        
        # Calculate the latest possible time the simulation can run for
        # For even number of spots this is N^2/2 - 1, for odd number of spots this is (N^2-1)/2 - 1
    
        if self.n_spots % 2 == 0:
            t_max = int((self.n_spots**2 / 2) - 1)
        else:
            t_max = int(((self.n_spots**2 - 1) / 2) - 1)
        
        #Create list of times
        _times = (np.arange(t_max + 1) * t_step)[np.newaxis, :]

    
        #Determine phase list from start_phase argument
        if starting_phase is None:
            #Use standard number of phases; here, 8
            _phases = np.linspace(0, 2*np.pi, 8, endpoint=False)[:, np.newaxis]
        else:
            _phases = np.array([starting_phase])[:, np.newaxis]

            
        #Create 4D tensor, currently filled with the phase shifts at each timepoint with each starting phase
        #Dimensions (P,1,T,1)
        _all_shifts = (self.amp*np.sin((_phase_speed*_times) + _phases))[:, np.newaxis, :, np.newaxis]

        #Find the left and right edges in dimensions (1, S, 1, 1)
        _left_edges = self.spot_edges[:-1][np.newaxis, :, np.newaxis, np.newaxis]
        _right_edges = self.spot_edges[1:][np.newaxis, :, np.newaxis, np.newaxis]
        
        #Reshape CTV space to (1,1,1,V)
        _ctv_space = self.ctv_space[np.newaxis, np.newaxis, np.newaxis, :]

        #Generate the masks for all phases, for all spots, for all time steps
        #Resulting tensor has dimensions (P,S,T,V)
        self.tensor = (_ctv_space > (_left_edges - _all_shifts)) & (_ctv_space < (_right_edges - _all_shifts))

        return self.tensor
    
    def evaluate_sequences(self, sequences):
        
        """
        Calculates MSE for a batch of sequences.
        Args:
            sequences: Array of sample sequences
        """
        
        n_sequences = sequences.shape[0]
        intended = self.get_intended_dist()
        
        #Calculate jumps and times for each sequence
        jumps = np.abs(np.diff(sequences, axis=1))
        times = np.zeros_like(sequences)
        times[:,1:] = np.cumsum(jumps, axis=1)
        
        #Generate empty array for all mses
        mses = np.zeros(n_sequences)
        
        #Loop through each sequence
        for sequence in range(n_sequences):
            
            # Find the dose distributions for each beam point at the correct time for all required phases    
            dists = self.tensor[:, sequences[sequence], times[sequence], :].sum(axis=1)
            
            # Find the squared differences between the distributions and intended across all phases
            # Still evaluates using MSE
            sq_diffs = (dists - intended)**2
            
            # Average across voxels and required phases
            mses[sequence] = np.mean(sq_diffs)
        
        return mses
    
    def display_sims(self, sims):
        
        '''
        Generate and display the intended target distribution, along with other simulated distributions
        '''
        
        _target_dist = self.get_intended_dist()
                
        for sim in sims:
            plt.plot(self.ctv_space, sim, lw=2, alpha= 0.9)
            
        plt.plot(self.ctv_space, _target_dist, 'k--', lw=2, label = "Intended Distribution")
            
        plt.ylabel('Relative Dose')
        plt.grid(True)
        plt.show()

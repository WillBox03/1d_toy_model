import numpy as np
import matplotlib.pyplot as plt

class PTV_env:   
    '''
    Environment space for the simulation
    '''
    def __init__(self, res, gtv_length, n_spots, amp):
        
        '''
        Initialises the environment, creating a high resolution coordinate space for the PTV, containing a target GTV of set length in the middle. Also defines positions of proton beam spots covering the PTV
        Args:
            res (float): Resolution for high coordinate space, in cm
            ptv_length (float,int): Length of GTV in coordinate space, in cm
            n_spots (int): Number of proton beam spots covering the PTV
            amp(float): Amplitude of breathing motion, in cm
        '''
        self.res = res
        self.n_spots = n_spots
        self.amp = amp # Include breathing amplitude to add surrounding tissue for PTV
        
        # Boundaries of GTV and PTV
        
        self.gtv_stop = gtv_length / 2
        self.gtv_start = -self.gtv_stop
        
        self.ptv_stop = self.gtv_stop + self.amp
        self.ptv_start = -self.ptv_stop
        
        self.num_voxels = int(np.round((self.ptv_stop - self.ptv_start) / self.res)) # Calculate number of voxels
        
        # Create hi-res PTV environment at voxel centres
        self.ptv_space = np.linspace(self.ptv_start + self.res/2, self.ptv_stop - self.res/2, self.num_voxels)

        # Define proton spot edges in environment
        self.spot_edges = np.linspace(self.ptv_start, self.ptv_stop, n_spots + 1)
        
        # Define GTV mask
        self.greater_gtv_mask = self.ptv_space > self.gtv_start
        self.less_gtv_mask = self.ptv_space < self.gtv_stop
        self.gtv_mask = np.logical_and(self.greater_gtv_mask, self.less_gtv_mask)


    def set_spot_weights(self, spot_weights):
        
        '''
        Defines the dosage weighting for each spot
        '''
        
        self.spot_weights = spot_weights


    def set_sequence(self, sequence):
        
        '''
        Defines the ordering sequence of the spots, with some preset order selections
        '''
        
        _presets = {'lr_rast' : np.arange(self.n_spots),
                    'rl_rast' : np.arange(self.n_spots - 1, -1, -1),
                    'rand' : np.random.permutation(self.n_spots)}
        
        if isinstance(sequence, str):
            if sequence in _presets:
                return _presets[sequence]
            else:
                raise ValueError(f"Unknown preset: '{sequence}'. Available: {list(_presets.keys())}")  
    
        return sequence

    def sim(self, timestep, freq, sequence, starting_phase):
        
        '''
        Simulates spot scanning sequence in a moving target, returning the resultant dose distribution
        Args:
            timestep (float): Time taken for proton beam to move to adjacent spots, in s
            freq (float): Frequency of breathing motion, in hz
            sequence (arr, str): Sequence for proton beam spots
        '''
        self.sequence = self.set_sequence(sequence)
        
        _phase = 2*np.pi*freq # Phase of breathing
        
        _jumps = abs(self.sequence[1:] - self.sequence[:-1]) # Distance between each proton beam spot in sequence
    
        # Cumulative time taken to move to each proton beam spot in sequence
        _times = np.zeros(len(_jumps) + 1)
        
        _times[1:] = np.cumsum(_jumps) * timestep

        _shifts = self.amp*np.sin((_phase*_times) - starting_phase) # Resultant positional shift at each time

        # Find the spot edges in sequence order
        _left_edges = self.spot_edges[self.sequence]
        _right_edges = self.spot_edges[self.sequence + 1]

        # Find spot edges relative to GTV position due to shifts  
        _left_shifts = _left_edges - _shifts
        _right_shifts = _right_edges - _shifts
        
        #Mask proton spots and GTV positions in PTV space
        _greater_than_masks = self.ptv_space > _left_shifts[:,None]
        _less_than_masks = self.ptv_space < _right_shifts[:,None]

        #Mask for proton spots in line with GTV
        _spot_masks = np.logical_and(_greater_than_masks, _less_than_masks)
        _dose_masks = np.logical_and(_spot_masks, self.gtv_mask)
        self.dose_dist = _dose_masks.T @ self.spot_weights

        return self.dose_dist
        

    def get_intended_dist(self):
        
        '''
        Deliver the intended distribution to PTV
        '''
        _greater_than_masks = self.ptv_space > self.spot_edges[self.sequence,None]
        _less_than_masks = self.ptv_space < self.spot_edges[self.sequence + 1,None]
        
        _spot_masks = np.logical_and(_greater_than_masks, _less_than_masks)

        _dose_masks = np.logical_and(_spot_masks, self.gtv_mask)
        
        _intended_dist = _dose_masks.T @ self.spot_weights
        
        return _intended_dist
    
    def sim_phases(self, timestep, freq, sequence):
        
        _phases = np.linspace(0, 2*np.pi, 8, endpoint=False)

        self.dose_dists = np.array([self.sim(timestep, freq, sequence, phase) for phase in _phases])

        return self.dose_dists
    
    def calc_mse(self, dose_dist):
        
        '''
        Calculates the mean squared error (mse) of a resultant dose distribution against the intended distribution
        '''
        
        _intended_dist = self.get_intended_dist()
        
        self.mse = np.mean((dose_dist - _intended_dist)**2)
        
        return self.mse
    
    def display_sims(self, sims):
        
        '''
        Generate and display the intended target distribution, along with other simulated distributions
        '''
        _target_dist = self.get_intended_dist()
                
        for sim in sims:
            plt.plot(self.ptv_space, sim, lw=2, alpha= 0.9)
            
        plt.plot(self.ptv_space, _target_dist, 'k--', lw=2, label = "Intended Distribution")
            
        plt.ylabel('Relative Dose')
        plt.grid(True)
        plt.show()
        
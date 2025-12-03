import numpy as np
import matplotlib.pyplot as plt

   
class PTV_env:   
    '''
    Environment space for the simulation
    '''
    def __init__(self, res, ptv_length, n_spots, amp):
        
        '''
        Initialises the environment, creating a high resolution coordinate space and defining the position and bounds proton spots which cover the ptv
        Args:
            res (float): Resolution for high coordinate space, in cm
            ptv_length (float,int): Length of PTV in coordinate space, in cm
            n_spots (int): Number of proton beam spots covering the ptv 
            amp(float): Amplitude of breathing motion, in cm
        '''
        
        self.n_spots = n_spots
        self.amp = amp # Include breathing amplitude to add surroundings tissue
        
        self.x_space = np.linspace(-((ptv_length/2)+self.amp), (ptv_length/2)+self.amp, int(1/res)) # Create hi-res environment
        
        #Define proton spot edges and centres in environment
        self.spot_edges = np.linspace(-ptv_length/2, ptv_length/2, n_spots + 1)        
        self.spot_centres = 0.5 * (self.spot_edges[:-1] + self.spot_edges[1:])


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
        
        if sequence in _presets:
            return _presets[sequence]
        else:
            return sequence   
    

    def sim(self, timestep, freq, sequence):
        
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
    
        # Cululative time taken to move to each proton beam spot in sequence
        _times = np.zeros(len(_jumps) + 1)
        
        _times[1:] = np.cumsum(_jumps) * timestep
        
        _shifts = self.amp*np.sin(_phase*_times) # Resultant positional shift at each time
        
        # Find the spot edges in resolution space and the motion shifted ptv position
        _left_edges = self.spot_edges[self.sequence]
        _right_edges = self.spot_edges[self.sequence + 1]
            
        _left_shifts = _left_edges - _shifts
        _right_shifts = _right_edges - _shifts
        
        #Mask points in shifted space and deliver spot-weighted distribution
        _greater_than_masks = self.x_space >= _left_shifts[:,None]
        _less_than_masks = self.x_space < _right_shifts[:,None]
        _masks = np.logical_and(_greater_than_masks, _less_than_masks)
        self.dose_dist = _masks.T @ self.spot_weights
        
        return self.dose_dist
        

    def get_intended_dist(self):
        
        '''
        Deliver the intended distribution to PTV
        '''
        _greater_than_masks = self.x_space >= self.spot_edges[self.sequence,None]
        _less_than_masks = self.x_space < self.spot_edges[self.sequence + 1,None]
        
        _masks = np.logical_and(_greater_than_masks, _less_than_masks)
        _intended_dist = _masks.T @ self.spot_weights
        
        return _intended_dist
    
    def display_sims(self, sims):
        
        '''
        Generate and display the intended target distribution, along with other simulated distributions
        '''
        _target_dist = self.get_intended_dist()
                
        for sim in sims:
            plt.plot(self.x_space, sim, lw=2, alpha= 0.9)
            
        plt.plot(self.x_space, _target_dist, 'k--', lw=2, label = "Intended Distribution")
            
        plt.ylabel('Relative Dose')
        plt.grid(True)
        plt.show()

  
  
resolution = 0.001 # in cm

ptv_length = 2 # in cm

n_spots = 10

amp = 0.5 #in cm 
 
ptv = PTV_env(resolution, ptv_length, n_spots, amp)

ptv.set_spot_weights(np.ones(n_spots))

t_step = 0.2 # in seconds

freq = 1/7 # in hertz

lr_rast_dist = ptv.sim(t_step, freq, 'lr_rast') # Left to right raster scan

rl_rast_dist = ptv.sim(t_step, freq, 'rl_rast') # Right to left raster scan

rand_dist = ptv.sim(t_step, freq, 'rand') # Random sequence scan

sims = np.array([lr_rast_dist, rl_rast_dist, rand_dist])

ptv.display_sims(sims)
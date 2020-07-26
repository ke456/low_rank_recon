import torch
import torch.nn as nn
from collections import namedtuple
import random
import pickle

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    """Data structure for storing episodes. Each episode
    is represented as a Transition.
    Parameters
    ----------
    capacity: int 
        Represents the size of the memory
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Adds transitions into memory. When the position index hits the end of memory,
        the position is reset back to the start of memory.
        Parameters
        ----------
        *args : list of *args of transitions
            Each transition is represented as [state, action, next_state, reward, done]
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Randomly sample transitions from memory.
        Parameters
        ----------
        batch_size : int
            Numer of transition to sample
        Returns
        -------
        transition : namedtuple
        """
        batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*batch))


    def __len__(self):
        return len(self.memory)

    
class BaseRLAgent(object):
    """ Base model which contains configuration properties preset for each model.
    Also provides a method to save the model using `from_pickle(filename)` and `to_pickle(filename)`.
    """
    
    def __init__(self,
            batch_size=256,
            max_steps=100,
            max_episodes=500,
            learning_rate=0.01,
            gamma=0.5,
            update_steps=100,
            epsilon=1,
            epsilon_decay=0.9,
            epsilon_min=0.01,
            optimizer=torch.optim.Adam,
            warm_start=False,
            device=None,
            verbose=1):
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_steps = update_steps
        self.optimizer = optimizer
        self.warm_start = warm_start
        self.verbose=verbose
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if verbose > 0:
                print("Using","cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.params = [
            'batch_size',
            'max_steps',
            'max_episodes,'
            'learning_rate',
            'gamma',
            'update_steps',
            'epsilon',
            'epsilon_decay',
            'epsilon_min'
            'optimizer',
            'l2_strength']
        self.errors = []
        self.dev_predictions = {}

    def to_pickle(self, output_filename):
        """Serialize the class instance. Will be phased out for torch.save later.
        Parameters
        ----------
        output_filename : str
            Full path for the output file.
        """
        self.model = self.model.cpu()
        with open(output_filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(src_filename):
        """Load an entire class instance onto the CPU.
        Parameters
        ----------
        src_filename : str
            Full path to the serialized model file.
        """
        self.warm_start = True
        with open(src_filename, 'rb') as f:
            return pickle.load(f)

    def __repr__(self):
        param_str = ["{}={}".format(a, getattr(self, a)) for a in self.params]
        param_str = ",\n\t".join(param_str)
        return "{}(\n\t{})".format(self.__class__.__name__, param_str)
    
    
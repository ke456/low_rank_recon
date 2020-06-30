import torch
import torch.nn as nn


class BaseRLAgent(object):
    """ Base model which contains configuration properties preset for each model.
    Also provides a method to save the model using `from_pickle(filename)` and `to_pickle(filename)`.
    """
    
    def __init__(self,
            batch_size=256,
            max_iter=100,
            learning_rate=0.001,
            gamma=0.99,
            update_steps=100,
            epsilon=1,
            epsilon_decay=0.9,
            epsilon_min=0.01,
            optimizer=torch.optim.Adam,
            l2_strength=0,
            warm_start=False,
            device=None,
            verbose=True):
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_steps = update_steps
        self.optimizer = optimizer
        self.l2_strength = l2_strength
        self.warm_start = warm_start
        self.verbose=verbose
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print("Using","cuda" if torch.cuda.is_available() else "cpu") if verbose
        self.device = torch.device(device)
        self.params = [
            'batch_size',
            'max_iter',
            'learning_rate',
            'gamma',
            'update_steps',
            'epsilon_decay',
            'epsilon_min'
            'optimizer',
            'l2_strength']
        self.errors = []
        self.dev_predictions = {}

    def to_pickle(self, output_filename):
        """Serialize the entire class instance. Importantly, this
        is different from using the standard `torch.save` method:
        torch.save(self.model.state_dict(), output_filename)
        The above stores only the underlying model parameters. In
        contrast, the current method ensures that all of the model
        parameters are on the CPU and then stores the full instance.
        This is necessary to ensure that we retain all the information
        needed to read new examples and make predictions.
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
        """Load an entire class instance onto the CPU. This also sets
        `self.warm_start = True` so that the loaded parameters are used
        if `fit` is called.
        Importantly, this is different from recommended PyTorch method:
        self.model.load_state_dict(torch.load(src_filename))
        We cannot reliably do this with new instances, because we need
        to see new examples in order to set some of the model
        dimensionalities and obtain information about what the class
        labels are. Thus, the current method loads an entire serialized
        class as created by `to_pickle`.
        The training and prediction code move the model parameters to
        `self.device`.
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
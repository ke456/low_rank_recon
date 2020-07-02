from model_abstract import ModelAbstract
import torch.nn as nn

class EstimatorModelBase(ModelAbstract):
    """The simple neural network model."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _initialize_layers(self):
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(256, self.output_dim)
    
    def _forward_input(self, env_state):
        return self.input_layer(env_state)
    
    def _forward_output(self, hidden_state):
        return self.output_layer(hidden_state)
    
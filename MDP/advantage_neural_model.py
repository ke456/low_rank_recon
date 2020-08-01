from model_abstract import ModelAbstract
import torch.nn as nn
from base_neural_model import EstimatorModelBase

class AdvantageModel(EstimatorModelBase):
    """The advantage neural network model."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _initialize_layers(self):
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        
        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, self.output_dim)
    
    def _forward_output(self, hidden_state):
        self.value_output = self.value(hidden_state)
        self.advantage_output = self.advantage(hidden_state)
        return self.advantage_output + self.value_output
    
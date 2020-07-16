import torch.nn as nn

class ModelAbstract(nn.Module):
    """An abstract class to extend neural network models.
    """
    
    def __init__(self, input_dim, output_dim):
        super(ModelAbstract, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._initialize_layers()
    
    def _initialize_layers(self):
        """This function defines the sequence of layers in the NN.
        """
        pass
    
    def _forward_input(self, env_state):
        """Defines the forward function for all layers except the last layer.
        """
        pass
    
    def _forward_output(self, hidden_state):
        """Defines the forward function for the last layer.
        """
        pass
    
    def forward(self, env_state):
        """Foward pass to predict the next action given state.
        Given that input and output
        """
        features = self._forward_input(env_state)
        features = features.view(features.size(0), -1)
        out = self._forward_output(features)
        return out
    

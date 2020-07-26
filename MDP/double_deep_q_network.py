from deep_q_network import DQNAgent

class DoubleDQNAgent(DQNAgent):
    """A Double Deep Q Network Agent
    """
    
    def __init__(self,
                 *args,
                 **kwargs):
        super(DoubleDQNAgent, self).__init__(*args, **kwargs)
    
    def _next_max_Q(self, next_states):
        """Returns the Q value of the target network using the online network's best action.
        
        Parameters
        ----------
        next_states: list of future states
        
        Returns
        -------
        List of floats, the max Q-values per each future state
        """
        _, next_state_actions = self.model.forward(next_states).max(1, keepdim=True)
        return self.target_model.forward(next_states).gather(1, next_state_actions)
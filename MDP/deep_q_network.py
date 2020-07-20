import torch
import torch.nn as nn
from base_rl_agent_torch import BaseRLAgent
import numpy as np
import random

class DQNAgent(BaseRLAgent):
    """Base model for Deep Q Network Agents
    Parameters
    ----------
        env: enviroment, be it openai's environment or our own
        model: Instance of a implemented ModelAbstract used for approximation
        replay_buffer: Replay Bufer to store Transitions
    """
    
    def __init__(self, 
                 env, 
                 model, 
                 replay_buffer,
                 loss=nn.MSELoss(),
                 exploration_penalty=-0.1,
                 **kwargs):
        super(DQNAgent, self).__init__(**kwargs)
        
        self.action_space_size = env.action_space
        self.observation_space_shape = env.observation_space
        
        self.replay_buffer = replay_buffer
        self.model = model(self.observation_space_shape, self.action_space_size)
        self.target_model = model(self.observation_space_shape, self.action_space_size)
        self.loss = loss    # This class owns the loss, not the model
        self.opt = self.optimizer(
                self.model.parameters(),
                lr=self.learning_rate)
        self.iteration = 0
        self.exploration_penalty = exploration_penalty
        
    def get_action(self, state, env):
        """ Epsilon greedy, with probability 1-epsilon of taking a greedy action, 
        otherwise of taking a random action.
        
        Parameters
        ----------
        state: vector which represents the state environment
        env: a object of the environment
              
        Returns
        -------
        action index in action space
        """
        if(np.random.uniform() < self.epsilon):
            return env.sample_action(state)
        
        mask = env.bool_feature(state[1])
        for i in range(len(mask)):
            if mask[i] == 1:
                mask[i] = float('-inf')
        
        mask = torch.FloatTensor(mask)
        state = torch.FloatTensor(state[1]).unsqueeze(0).to(self.device)
        
        
        q_values = self.model.forward(state)
        
        q_values = q_values + mask
        
        
        return np.argmax(q_values.cpu().detach().numpy()) # Detaches to prevent unused gradient flow

    def compute_loss(self, batch):
        """ Calculates loss of a batch using loss.
        
        Parameters
        ----------
        batch: n by m matrix of Transitions
              
        Returns
        -------
        loss of the TD error (predicted value function and gamma*value function(next_state)).
        """

        states, actions, rewards, next_states, done = self.__get_tensors_from_batch__(batch)

        q_predictions = self._predict_q(states, actions)

        # Get the TD target as the target estimate of Q
        target_Q = self._TD_target(next_states, done, rewards)
        
        loss = self.loss(q_predictions, target_Q.detach())
        
        return loss

    def _TD_target(self, next_states, done, rewards):
        """Returns the TD target for the transition
        Parameters
        ----------
        next_states: list of future states
        done: list of boolean
        rewards: list of float
        
        Returns
        -------
        A list of floats which represent the TD target for each transition in the batch
        """
        max_next_Q = self._next_max_Q(next_states)
        td_target = rewards + (1 - done) * self.gamma * max_next_Q
        return td_target

    def train(self, env, max_episodes, max_steps, batch_size):
        """Trains a Deep Q Agent for the given environment.
        Parameters
        ----------
        env: Environment model
            A simulation of the environment which implements reset()
            and step(action).
        max_episodes: int
        max_steps: int
        batch_size: int
        """
        episode_rewards = []

        for episode in range(max_episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                #print(state)
                action = self.get_action(state, env)
                #print(action)
                next_state, reward, done, _ = env.step(state, action)
                #print(reward)
                reward += self.exploration_penalty
                self.replay_buffer.push(state[1], action, next_state, reward, done)
                episode_reward += reward

                if len(self.replay_buffer) > batch_size:
                    self.update(batch_size)
                    self.iteration += 1

                if done or step == max_steps-1:
                    episode_rewards.append(episode_reward)
                    print("Episode " + str(episode) + ": " + str(episode_reward), '\t', self.epsilon)
                    self.update_epsilon()
                    break

                state = next_state

        return episode_rewards

    def _predict_q(self, states, actions):
        """Value function for state action pairs.
        Parameters
        ----------
        states : list of states
        actions: list of actions
        
        Returns
        -------
        list of Q values for each actions/state pair
        """
        return self.model.forward(states).gather(1, actions).to(self.device)

    def __get_tensors_from_batch__(self, batch):
        """Maps transitions in a batch to tensors.
        Parameters
        ----------
        batch: list of Transitions
        
        Returns
        -------
        float tensors of Transitions
        """
        states = torch.FloatTensor(batch.state, device=self.device)
        actions = torch.LongTensor(batch.action, device=self.device)
        rewards = torch.FloatTensor(batch.reward, device=self.device)
        next_states = torch.FloatTensor([nextstate[1] for nextstate in batch.next_state], device=self.device)
        done = torch.FloatTensor(batch.done, device=self.device)
        
        # Unsqueeze a dimension
        actions = actions.view(actions.size(0), 1)
        done = done.view(done.size(0), 1)
        rewards = rewards.view(rewards.size(0), 1)

        return states, actions, rewards, next_states, done

    def _next_max_Q(self, next_states):
        """Returns the max Q value for a batch of future states.
        Parameters
        ----------
        next_states: list of future states
        
        Returns
        -------
        List of floats, the max Q-values per each future state
        """
        next_Q = self.target_model.forward(next_states).to(self.device)
        max_next_Q = torch.max(next_Q, 1)[0]
        return max_next_Q.view(max_next_Q.size(0), 1)

    def update(self, batch_size):
        """Forward and backward pass for a batch. Also updates the target network with the
        moving network after a certain number of iterations.
        
        Parameters
        ----------
        batch_size: int
            Number of Transitions to be sampled from memory
        """

        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Set the moving network to the target network
        if self.iteration % self.update_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


def simple_example():
    from base_rl_agent_torch import ReplayMemory
    from base_neural_model import EstimatorModelBase
    import gym
    import utils
    # Fix seeds for reproducibility. Note that OpenAI's environment has
    # stocastic starting states so the resulting model will be different, anyway
    utils.fix_seeds()
    
    env_id = "CartPole-v1" # OpenAI's environment name
    MAX_EPISODES = 100
    MAX_STEPS = 500
    BATCH_SIZE = 32
    
    # Define the environment
    env = gym.make(env_id)
    buffer = ReplayMemory(10000)
    # Initiate the agent
    model = EstimatorModelBase
    agent = DQNAgent(env, model, buffer)
    
    #Train
    episode_scores = agent.train(env, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)
    
    # Now we evaluate the trained model by taking greedy actions
    observation = env.reset() # Current observed state
    total_reward = 0
    agent.epsilon = 0
    while True:
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done: 
            break;
    print("Test Total Reward:", total_reward)
    
    # Saving the network
    agent.to_pickle("somefile.pickle")
        
if __name__ == '__main__':
    simple_example()
    
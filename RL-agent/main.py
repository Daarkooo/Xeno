import numpy as np
import pandas as pd
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import time  # Import time module for tracking training duration

# Load synthetic data for training
train_data = pd.read_csv('train_data.csv', sep=';')  # Ensure to parse the date
test_data = pd.read_csv('test_data.csv', sep=';')

train_data['Date'] = pd.to_datetime(train_data['Date'], format='%d/%m/%Y %H:%M')
test_data['Date'] = pd.to_datetime(test_data['Date'], format='%d/%m/%Y %H:%M')

# Constants
TOTAL_BANDWIDTH = 10000  # Total bandwidth in Kbps (10 Mbps)
MIN_BANDWIDTH = 1000      # Minimum bandwidth per user in Kbps (CIR)
NUM_USERS = 10            # Total number of users
TIME_STEPS = 288          # Total time steps for 24 hours (5 mins each)
THRESHOLD_EXCEEDANCE = 0.2 # 20%
MIN_DURATION = 3           # Minimum duration for abusive usage
PENALTY_COEFFICIENT_OVER = 3
PENALTY_COEFFICIENT_ABUSIVE = -0.5

# Define the environment for bandwidth allocation
class BandwidthAllocationEnv(gym.Env):
    def __init__(self, data):
        super(BandwidthAllocationEnv, self).__init__()
        self.data = data.copy()  # Create a copy of the DataFrame
        self.current_step = 0
        self.num_users = NUM_USERS
        
        # Initialize MIRs (maximum information rate for each user)
        self.mirs = np.full(NUM_USERS, MIN_BANDWIDTH)  # Set initial MIRs to MIN_BANDWIDTH
        
        # Define action space (adjust MIRs)
        self.action_space = spaces.Box(
            low=MIN_BANDWIDTH * np.ones(NUM_USERS), 
            high=(TOTAL_BANDWIDTH - (NUM_USERS - 1) * MIN_BANDWIDTH) * np.ones(NUM_USERS),  # Ensure sum of MIRs ≤ Remaining Bandwidth
            dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(NUM_USERS, 5),  # (MIR, Requested, Allocated, Abuse Indicator, Time of Day)
            dtype=np.float32
        )
        
        # Initialize other relevant attributes
        self.requests = np.zeros((NUM_USERS, TIME_STEPS))  # Placeholder for bandwidth requests
        self.allocated_bandwidth = np.zeros(NUM_USERS)  # Tracks allocated bandwidth for each user
        
    def reset(self):
      self.current_step = 0
      self.abuse_counters = np.zeros(self.num_users)  # Reset abuse counters for each user
      self.abuse_flags = np.zeros(self.num_users, dtype=bool)  # Reset abuse flags for each user
      
      # Perform the initial allocation for each user
      self._initial_allocation()  # Call the initial allocation method

      return self._get_observation()  # Return the initial observation after allocation

    
    def _get_observation(self):
      observation = []
      for user_id in range(self.num_users):
          # Get the current bandwidth requested by this user
          user_data = self.data[self.data['DID'] == user_id]  # Filter based on DID
          requested = user_data.iloc[self.current_step]['BW_REQUESTED']
          
          # Calculate other necessary data (allocated, abuse indicator, MIR)
          allocated = self._calculate_allocated_bandwidth(user_id)
          abuse_indicator = 1 if self.abuse_flags[user_id] else 0
          mir = self.action_space.low[user_id]
          observation.append([mir, requested, allocated, abuse_indicator])
      return np.array(observation)
    
    def _calculate_allocated_bandwidth(self, user_id):
      requested = self.data[self.data['DID'] == user_id].iloc[self.current_step]['BW_REQUESTED']
      return min(requested, TOTAL_BANDWIDTH / self.num_users)

    
    def step(self, action):
      self.current_step += 1

      # Clip the action to ensure it meets MIR constraints
      action = np.clip(action, MIN_BANDWIDTH, TOTAL_BANDWIDTH)

      # Phase 1 - Initial Allocation
      self._initial_allocation()

      remaining_bandwidth = TOTAL_BANDWIDTH - np.sum(self.allocations)


      # Phase 2 - Apply MIR adjustments
      self._adjust_mirs(action)

      # Calculate reward
      reward = self._calculate_reward()

      # Check if the end of data is reached
      done = self.current_step >= len(self.data) - 1

      return self._get_observation(), reward, done, {}

    
    def _initial_allocation(self):
      self.allocations = np.zeros(self.num_users)  # Initialize an array to hold allocations for each user
      for user_id in range(self.num_users):
          # Get the requested bandwidth for the current user at the current step
          requested = self.data[self.data['DID'] == user_id].iloc[self.current_step]['BW_REQUESTED']
          
          # Ensure Minimum Bandwidth (CIR)
          if requested >= MIN_BANDWIDTH:
              self.allocations[user_id] = MIN_BANDWIDTH  # Allocate 1 Mbps (1000 Kbps)
          else:
              self.allocations[user_id] = requested  # Allocate exactly what they request


    
    def _adjust_mirs(self, action):
        # Phase 2 - RL Agent Optimizes Remaining Bandwidth
        # Adjust MIRs based on the action provided
        for user_id in range(self.num_users):
            self.action_space.low[user_id] = action[user_id]
    
    def _calculate_reward(self):
      efficiency_reward = 0
      over_allocation_penalty = 0
      total_allocated = 0

      # Initialize aggregate abuse score
      S_total = 0

      for user_id in range(self.num_users):
          requested = self.data[self.data['DID'] == user_id].iloc[self.current_step]['BW_REQUESTED']
          allocated = self.allocations.get(user_id, 0)
          total_allocated += allocated

          # Calculate Allocation Ratio based on request vs. MIR
          if requested >= self.action_space.low[user_id]:  # Requested Bandwidth ≥ MIR
              allocation_ratio = self.action_space.low[user_id] / requested
              efficiency_reward += allocation_ratio
          else:  # Requested Bandwidth < MIR
              efficiency_reward += 1  # Full satisfaction

          # Over-Allocation Penalty
          if total_allocated > 10000:  # TOTAL_BANDWIDTH as 10,000 Kbps
              over_allocation_penalty += 3 * (total_allocated - 10000) / 10000  # Penalty coefficient β = 3

          # Abusive Usage Detection and Penalty
          if requested > self.action_space.low[user_id] * 1.2:  # Exceeding MIR by threshold θ = 20%
              self.abuse_counters[user_id] += 1
              if self.abuse_counters[user_id] >= 3:  # Minimum duration Δt_min = 3
                  # Increment S_total for each consecutive abusive episode beyond Δt_min
                  S_total += (self.abuse_counters[user_id] - 3 + 1)  # Abuse duration - Δt_min + 1
          else:
              self.abuse_counters[user_id] = 0

      # Calculate the penalty for abusive usage
      abusive_usage_penalty = -0.5 * S_total / (self.num_users * self.total_time_steps)  # Penalty coefficient γ = -0.5

      # Total Reward Calculation
      total_reward = (efficiency_reward / self.num_users) - over_allocation_penalty - abusive_usage_penalty
      return total_reward


# Define the PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.lamda = 0.95
        self.learning_rate = 0.001
        self.policy_net = self._build_network()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )

    def act(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            action = self.policy_net(state).numpy()
        return action + np.random.normal(0, 0.1, size=self.action_dim)  # Add noise for exploration

    def update(self, states, actions, rewards, next_states, dones):
        # Implement the PPO algorithm here (this is a placeholder)
        # Here you would add the logic for updating the policy using PPO
        pass

# Training the agent
def train_agent(env, agent, episodes):
    start_time = time.time()  # Start time tracking
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state.flatten())
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    end_time = time.time()  # End time tracking
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

# Instantiate environment and agent
env = BandwidthAllocationEnv(train_data)
agent = PPOAgent(state_dim=NUM_USERS * 4, action_dim=NUM_USERS)

# Start training
train_agent(env, agent, episodes=1000)

# Test the agent
def test_agent(env, agent):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state.flatten())
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
    print(f"Total Reward in Testing: {total_reward}")

# Test the trained agent
test_agent(env, agent)


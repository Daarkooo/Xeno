import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

class SatelliteBandwidthEnv(gym.Env):
    def __init__(self, data, total_bandwidth=10000, cir=1000):
        super(SatelliteBandwidthEnv, self).__init__()
        self.data = data
        self.num_users = 10  # Fixed number of users
        self.total_bandwidth = total_bandwidth
        self.cir = cir
        self.state = np.zeros((self.num_users, 5))
        self.time_step = 0  # Initialize time step

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_users, 5), dtype=np.float32)
        self.action_space = spaces.Box(low=cir, high=total_bandwidth, shape=(self.num_users,), dtype=np.float32)

    def reset(self, time_step=0):
        self.time_step = time_step
        self.state = np.zeros((self.num_users, 5))
        self.state[:, 0] = self.cir  # Set initial MIR to CIR for each user

        # Get requested bandwidths for the current time step
        requested_bandwidths = self.data.iloc[self.time_step * self.num_users:(self.time_step + 1) * self.num_users]['BW_REQUESTED'].values
        self.state[:, 1] = requested_bandwidths
        
        return self.state

    def step(self, action):
        for i in range(self.num_users):
            requested_bandwidth = self.state[i, 1]
            if requested_bandwidth >= self.cir:
                self.state[i, 2] = self.cir  # Allocate CIR minimum bandwidth
            else:
                self.state[i, 2] = requested_bandwidth
        
        mir_adjustments = np.clip(action, self.cir, self.total_bandwidth)
        total_allocated_bandwidth = np.sum(mir_adjustments)

        if total_allocated_bandwidth <= self.total_bandwidth:
            for i in range(self.num_users):
                self.state[i, 0] = mir_adjustments[i]
                self.state[i, 2] += mir_adjustments[i]

        reward = self._calculate_reward()
        self.time_step += 1  # Move to the next time step
        done = self.time_step >= (len(self.data) // self.num_users)  # Check if time steps exhausted
        return self.state, reward, done, {}

    def _calculate_reward(self):
        efficiency_reward = 0
        for i in range(self.num_users):
            requested_bandwidth = self.state[i, 1]
            allocated_bandwidth = self.state[i, 2]
            if requested_bandwidth >= self.state[i, 0]:
                efficiency_reward += allocated_bandwidth / requested_bandwidth
            else:
                efficiency_reward += 1  # Fully satisfied request
        
        over_allocation_penalty = 0
        abusive_usage_penalty = 0
        total_allocated = np.sum(self.state[:, 2])
        if total_allocated > self.total_bandwidth:
            over_allocation_penalty = (total_allocated - self.total_bandwidth) * 3
        
        for i in range(self.num_users):
            requested_bandwidth = self.state[i, 1]
            mir = self.state[i, 0]
            if requested_bandwidth > mir * 1.2:
                abuse_flag = self.state[i, 3]
                self.state[i, 3] = abuse_flag + 1
                if self.state[i, 3] >= 3:
                    abusive_usage_penalty += 1
        
        total_penalty = over_allocation_penalty + abusive_usage_penalty
        efficiency_reward /= self.num_users  # Normalize
        return efficiency_reward - total_penalty


# Load your train and test data
train_df = pd.read_csv('train_data.csv',sep=';')
test_df = pd.read_csv('test_data.csv',sep=';')  

train_df['Date'] = pd.to_datetime(train_df['Date'], format='%d/%m/%Y %H:%M')
test_df['Date'] = pd.to_datetime(test_df['Date'], format='%d/%m/%Y %H:%M')

train_df['Hour'] = train_df['Date'].dt.hour
test_df['Hour'] = test_df['Date'].dt.hour

hourly_usage = train_df.groupby('Hour')['BW_REQUESTED'].sum()



# Initialize the environment for training
env = SatelliteBandwidthEnv(data=train_df)

# Train the PPO model
model = PPO("MlpPolicy", env, learning_rate=0.001, n_steps=256, verbose=1)
model.learn(total_timesteps=50000)

# Save the model after training
model.save("ppo_satellite_bandwidth")

# Load the trained model (optional if you've already trained)
model = PPO.load("ppo_satellite_bandwidth")

# Test the trained agent with test data
test_env = SatelliteBandwidthEnv(data=test_df)
obs = test_env.reset()git
for i in range(100):
    action, _states = model.predict(obs)
    obs, reward, done, info = test_env.step(action)
    print(f"Step {i} - Reward: {reward}")

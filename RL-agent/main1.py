import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from datetime import datetime

class SatelliteBandwidthEnv(gym.Env):
    def __init__(self, data, total_bandwidth=10000, cir=1000):
        super(SatelliteBandwidthEnv, self).__init__()
        self.data = data
        self.num_users = 10
        self.total_bandwidth = total_bandwidth
        self.cir = cir
        self.state = np.zeros((self.num_users, 7))
        self.time_step = 0

        self.R_efficiency_avg=0
        self.N=0

        # Penalty coefficients
        self.penalty_coefficient_over = 3
        self.penalty_coefficient_abusive = -0.5
        self.abuse_threshold = 0.2
        self.min_duration = 3
        # Initialize abuse counters
        self.abuse_counters = np.zeros(self.num_users)  # Abuse counters for each user

        # Define observation and action spaces
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_users, 7), dtype=np.float32)
        self.action_space = spaces.Box(low=cir, high=total_bandwidth, shape=(self.num_users,), dtype=np.float32)

    def reset(self, time_step=0):
        # Initialize time step and state
        self.time_step = time_step
        self.state = np.zeros((self.num_users, 7))

        # Extract data for the current time step
        current_data = self.data.iloc[self.time_step * self.num_users:(self.time_step + 1) * self.num_users]

        # Set Device ID (DID) as integers
        self.state[:, 0] = current_data['DID'].astype(int).values
        print("DID:", self.state[:, 0].astype(int))

        # Convert 'Date' to datetime format if it's in string format
        if current_data['Date'].dtype == 'object':
            current_data['Date'] = pd.to_datetime(current_data['Date'])

        # Extract Unix timestamp directly or use hour for usage patterns
        self.state[:, 1] = current_data['Date'].apply(lambda x: x.timestamp()).values  # Unix timestamp

        # Set Bandwidth Requested
        self.state[:, 2] = current_data['BW_REQUESTED'].values

        # Initialize Bandwidth Allocated to zero
        self.state[:, 3] = self.cir

        # Set CIR for all users
        self.state[:, 4] = self.cir

        # Initialize MIR to CIR initially
        self.state[:, 5] = self.cir

        # Initialize Abuse Usage Indicator to zero
        self.state[:, 6] = 0

        return self.state
    import numpy as np

    def step(self, action):
        # Phase 1: Initial Allocation (Ensure Minimum Bandwidth - CIR)
        initial_allocations = np.zeros(self.num_users)
        
        # Allocate CIR or requested bandwidth if less than CIR
        for i in range(self.num_users):
            requested_bw = self.state[i, 2]  # Bandwidth Requested
            initial_allocations[i] = min(requested_bw, self.state[i, 4])  # CIR in self.state[:, 4]
        
        # Update the allocated bandwidth in the state (Phase 1 allocations)
        self.state[:, 3] = initial_allocations  # Bandwidth Allocated after Phase 1
        
        # Calculate remaining bandwidth after CIR allocation
        remaining_bandwidth = self.total_bandwidth - np.sum(initial_allocations)
        
        # Phase 2: RL Agent Optimizes Remaining Bandwidth by Adjusting MIR
        # Clip the action to be within CIR and the remaining bandwidth limits
        mir_adjustments = np.clip(action, self.state[:, 4], remaining_bandwidth)  # CIR as lower limit
        # print('MIRRRRRRRRRRRRRRRRRRRRRR : ',mir_adjustments)

        # Scale down MIRs if their sum exceeds the remaining bandwidth
        if np.sum(mir_adjustments) > remaining_bandwidth:
            mir_adjustments *= remaining_bandwidth / np.sum(mir_adjustments)
        # print('MIRRRRRRRRRRRRRRRRRRRRRR 2222222 : ',mir_adjustments)

        # Update MIR in the state after RL agent’s action
        self.state[:, 5] = mir_adjustments  # MIR after RL adjustment
        
        # Final allocation after RL agent's adjustments
        for i in range(self.num_users):
            additional_alloc = min(self.state[i, 2] - initial_allocations[i], mir_adjustments[i] - initial_allocations[i])
            self.state[i, 3] = initial_allocations[i] + max(0, additional_alloc)
        
        # Ensure total allocation does not exceed 10,000 Kbps
        total_allocated = np.sum(self.state[:, 3])
        if total_allocated > self.total_bandwidth:
            scaling_factor = self.total_bandwidth / total_allocated
            self.state[:, 3] *= scaling_factor
        
        # Calculate the reward and update time step
        reward = self.calculate_reward()
        self.time_step += 1
        done = self.time_step >= (len(self.data) // self.num_users)
        
        return self.state, reward, done, {}

    def calculate_reward(self):
            """Calculates the total reward at each time step."""
            R_efficiency = self.calculate_efficiency_reward()
            P_over = self.calculate_over_allocation_penalty()
            P_abusive = self.calculate_abusive_usage_penalty(288)
            self.R_efficiency_avg += R_efficiency
            self.N += 1
            
            # Total reward calculation
            R_total = R_efficiency - P_over - P_abusive
            return R_total

    def calculate_efficiency_reward(self):
            """Calculates the bandwidth allocation efficiency reward."""
            efficiency_scores = []
            
            for i in range(self.num_users):
                requested_bandwidth = self.state[i, 2]  # BW requested
                allocated_bandwidth = self.state[i, 3]  # BW allocated
                mir = self.state[i, 5]  # MIR
                
                if requested_bandwidth >= mir:
                    r_i_t = mir / requested_bandwidth if requested_bandwidth > 0 else 0
                else:
                    r_i_t = 1  # Full satisfaction
                
                efficiency_scores.append(r_i_t)
            
            R_efficiency = np.mean(efficiency_scores)
            return R_efficiency

    def calculate_over_allocation_penalty(self):
        """Calculates the penalty for over-allocation of total bandwidth."""
        total_allocated = np.sum(self.state[:, 3])  # Sum of allocated bandwidths
        penalty = max(0, total_allocated - self.total_bandwidth)  # Over-allocation amount
        
        P_over = self.penalty_coefficient_over * (penalty / self.total_bandwidth) if penalty > 0 else 0
        return P_over

    def calculate_abusive_usage_penalty(self, total_time_steps):
        """Calculates the penalty for sustained abusive usage."""
        total_abuse_score = 0
        abuse_durations = np.zeros(self.num_users)  # To track duration of abusive usage

        for i in range(self.num_users):
            requested_bandwidth = self.state[i, 2]  # BW requested
            mir = self.state[i, 5]  # MIR

            # Check if the user is in an abusive state
            if requested_bandwidth > mir * (1 + self.abuse_threshold):
                self.abuse_counters[i] += 1  # Increment abuse counter
            else:
                # Reset abuse counter if condition is not met
                if self.abuse_counters[i] >= self.min_duration:
                    total_abuse_score += (self.abuse_counters[i] - self.min_duration)  # Add to total score
                self.abuse_counters[i] = 0  # Reset counter

        # Handle remaining counters for users in abusive state at the end
        for i in range(self.num_users):
            if self.abuse_counters[i] >= self.min_duration:
                total_abuse_score += (self.abuse_counters[i] - self.min_duration)

        # Calculate total abuse score across users
        S_total = total_abuse_score

        # Calculate penalty
        P_abusive = self.penalty_coefficient_abusive * (S_total / (self.num_users * total_time_steps))
        return P_abusive
    


# Load your test data
train_df = pd.read_csv('optim_train_set.csv', sep=',')  # Adjust the path as necessary

train_df = train_df.sort_values('Date')

# Initialize the testing environment with the test dataset
test_env = SatelliteBandwidthEnv(data=train_df)

# Load the trained model
model = PPO.load("ppo_satellite_bandwidth")

# Reset the environment to start a new episode
obs = test_env.reset(time_step=0)

print("Testing the trained agent...\n")

#NUM_EPOCHS = 10

# Run a simple episode and print rewards for a specified number of steps
#for epoch in range(NUM_EPOCHS):  # Change the number of epochs if needed
obs = test_env.reset(time_step=0)  # Reset the environment for a new episode
for i in range(len(train_df)//10):  # Change the number of steps if needed
    action, _states = model.predict(obs[:, :5])  # Predict action from the model
    obs, reward, done, info = test_env.step(action)  # Take action in the environment
    print(f"Step {i} - Reward: {reward:.2f}")
    # average effiency

    # Check if the episode is done
    if done:
        print("Episode finished!")
        break

print(test_env.R_efficiency_avg/test_env.N)







# Load and preprocess your test data
test_df = pd.read_csv('test_data.csv', sep=';')  # Use the correct delimiter ';'
print(test_df)
# Convert 'Date' column to datetime and then to Unix timestamp
test_df['Date'] = pd.to_datetime(test_df['Date'], format='%d/%m/%Y %H:%M')
test_df = test_df.sort_values('Date')  # Sort by date if necessary

# Initialize the testing environment with the processed test dataset
test_env = SatelliteBandwidthEnv(data=test_df)

# Load the trained model
model = PPO.load("ppo_satellite_bandwidth")

# Reset the environment to start a new episode
obs = test_env.reset(time_step=0)

print("Testing the trained agent...\n")

# Run an episode and print rewards for a specified number of steps
for i in range(1000):  # Adjust number of steps if needed
    action, _states = model.predict(obs[:, :5])  # Predict action from the model
    obs, reward, done, info = test_env.step(action)  # Take action in the environment
    print(f"Step {i} - Reward: {reward:.2f}")

    # Check if the episode is done
    if done:
        print("Episode finished!")
        break

# Calculate and print average efficiency
print("Average Efficiency:", test_env.R_efficiency_avg / test_env.N)

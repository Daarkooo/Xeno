import numpy as np
import pandas as pd
from gym import Env, spaces
from stable_baselines3 import PPO

train_data = pd.read_csv('train_data.csv',sep=',')
test_data = pd.read_csv('test_data.csv',sep=';')



class BandwidthAllocationEnv(Env):
    def __init__(self, data, system_capacity=10000, num_users=10, cir=1000, threshold_exceedance=0.2, min_duration=3, penalty_coefficient_over=3, penalty_coefficient_abusive=-0.5):
        self.data = data
        self.system_capacity = system_capacity
        self.num_users = num_users
        self.cir = cir
        self.time_steps = 288  # 24 hours with 5-minute intervals
        self.threshold_exceedance = threshold_exceedance
        self.min_duration = min_duration
        self.penalty_coefficient_over = penalty_coefficient_over
        self.penalty_coefficient_abusive = penalty_coefficient_abusive
        
        # Action space: Adjust MIR for each user
        self.action_space = spaces.Box(low=1000, high=system_capacity, shape=(num_users,), dtype=np.int32)
        
        # State space: Current MIR, Bandwidth Requested, Bandwidth Allocated, Abusive Usage Indicator, Time of Day
        self.observation_space = spaces.Box(low=0, high=system_capacity, shape=(num_users, 5), dtype=np.float32)
        
        # Initialize state
        self.reset()
        
    def reset(self):
        # Reset environment state
        self.current_step = 0
        self.current_user_data = self.data.iloc[self.current_step]
        self.mir = np.ones(self.num_users) * self.cir
        self.bandwidth_requested = np.repeat(self.current_user_data['BW_REQUESTED'], self.num_users)
        self.bandwidth_allocated = self._allocate_bandwidth()
        self.abusive_usage_counter = np.zeros(self.num_users, dtype=int)
        self.abusive_usage_duration = np.zeros(self.num_users, dtype=int)
        self.time_of_day = self.current_step % (self.time_steps // 24)
        
        return self._get_state()
    
    def step(self, actions):
        # Update state based on actions
        self.mir = np.clip(actions, self.cir, self.system_capacity)
        self.current_step += 1
        self.current_user_data = self.data.iloc[self.current_step]
        self.bandwidth_requested = np.repeat(self.current_user_data['BW_REQUESTED'], self.num_users)
        self.bandwidth_allocated = self._allocate_bandwidth()
        self._update_abusive_usage()
        self.time_of_day = self.current_step % (self.time_steps // 24)
        
        # Calculate rewards
        r_efficiency = self._calculate_efficiency_reward()
        p_over = self._calculate_over_allocation_penalty()
        p_abusive = self._calculate_abusive_usage_penalty()
        reward = r_efficiency - p_over - p_abusive
        
        # Check if episode is done
        done = self.current_step == len(self.data) - 1
        
        return self._get_state(), reward, done, {'efficiency_reward': r_efficiency, 'over_allocation_penalty': p_over, 'abusive_usage_penalty': p_abusive}
    
    def _get_state(self):
        return np.column_stack((self.mir, self.bandwidth_requested, self.bandwidth_allocated, self.abusive_usage_counter > 0, np.repeat(self.time_of_day, self.num_users)))
    
    def _allocate_bandwidth(self):
        # Phase 1: Ensure Minimum Bandwidth (CIR)
        bandwidth_allocated = np.zeros(self.num_users)
        for i in range(self.num_users):
            if self.bandwidth_requested[i] >= self.cir:
                bandwidth_allocated[i] = self.cir
            else:
                bandwidth_allocated[i] = self.bandwidth_requested[i]
        
        # Phase 2: RL Agent Optimizes Remaining Bandwidth
        remaining_bandwidth = self.system_capacity - np.sum(bandwidth_allocated)
        
        for i in range(self.num_users):
            if self.bandwidth_requested[i] >= self.mir[i]:
                bandwidth_allocated[i] = min(self.mir[i], self.bandwidth_requested[i])
            else:
                bandwidth_allocated[i] += min(self.bandwidth_requested[i] - bandwidth_allocated[i], self.mir[i] - bandwidth_allocated[i])
        
        # Ensure total allocations do not exceed system capacity
        if np.sum(bandwidth_allocated) > self.system_capacity:
            bandwidth_allocated = bandwidth_allocated * (self.system_capacity / np.sum(bandwidth_allocated))
        
        return bandwidth_allocated
    
    def _calculate_efficiency_reward(self):
        # Implement efficiency reward calculation based on the provided formula
        efficiency_reward = 0
        for i in range(self.num_users):
            if self.bandwidth_requested[i] >= self.mir[i]:
                efficiency_reward += self.mir[i] / self.bandwidth_requested[i]
            else:
                efficiency_reward += 1
        return efficiency_reward / self.num_users
    
    def _calculate_over_allocation_penalty(self):
        # Implement over-allocation penalty calculation based on the provided formula
        over_allocation = max(0, np.sum(self.bandwidth_allocated) - self.system_capacity)
        return self.penalty_coefficient_over * (over_allocation / self.system_capacity)
    
    def _calculate_abusive_usage_penalty(self):
        # Implement abusive usage penalty calculation based on the provided formula
        total_abuse_score = 0
        for i in range(self.num_users):
            if self.bandwidth_requested[i] > self.mir[i] * (1 + self.threshold_exceedance):
                self.abusive_usage_counter[i] += 1
            else:
                self.abusive_usage_counter[i] = 0
            
            if self.abusive_usage_counter[i] >= self.min_duration:
                self.abusive_usage_duration[i] = self.abusive_usage_counter[i] - self.min_duration
                total_abuse_score += self.abusive_usage_duration[i]
            else:
                self.abusive_usage_duration[i] = 0
        
        return self.penalty_coefficient_abusive * (total_abuse_score / (self.num_users * self.time_steps))
    
    def _update_abusive_usage(self):
        # No need to update abusive usage indicator, as it is already handled in the _calculate_abusive_usage_penalty method
        pass

def evaluate_model(data):
    env = BandwidthAllocationEnv(data)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=288)
    
    # Evaluation
    print("Evaluation Rewards by Step:")
    obs = env.reset()
    done = False
    total_reward = 0
    total_efficiency_reward = 0
    total_over_allocation_penalty = 0
    total_abusive_usage_penalty = 0
    step_rewards = []
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        
        total_reward += reward
        total_efficiency_reward += info['efficiency_reward']
        total_over_allocation_penalty += info['over_allocation_penalty']
        total_abusive_usage_penalty += info['abusive_usage_penalty']
        step_rewards.append(reward)
        
    
    print(f"Total Reward: {total_reward}")
    print(f"Total Efficiency Reward: {total_efficiency_reward}")
    print(f"Total Over-Allocation Penalty: {total_over_allocation_penalty}")
    print(f"Total Abusive Usage Penalty: {total_abusive_usage_penalty}")
    print(f"Step Rewards: {step_rewards}")

# Example usage
evaluate_model(train_data)
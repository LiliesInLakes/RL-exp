import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
# from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

# from gymnasium.wrappers.monitoring import video_recorder


render_mode= None
env= gym.make("BipedalWalker-v3", render_mode= "rgb_array")
episodes= 10

env = RecordVideo(
    env, 
    video_folder="bipedal_vid", 
    episode_trigger=lambda episode_id: episode_id % 5 == 0
)
observation, info= env.reset()
print(f"observations are:{observation}")
# cart_pos, cart_vel, pole_angle, pole_w= observation

state_dim = env.observation_space.shape[0] 
action_dim = env.action_space.n


model= PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=episodes)
learning_rate= 0.001
optimizer = optim.Adam(model.parameters(), learning_rate)
score=0
done= False
best_score=0
gamma = 0.99
batch_size=5
print(f"actions are {env.action_space}")
total_rewards= []
total_disc= []
total_probs = []
for i in range(0, episodes):
    observation, _ = env.reset()
    done= False
    score= 0
    rewardlist= []
    log_probs= []
    while not done: #agent
        input_tensor= torch.FloatTensor(observation)
        logits= model(input_tensor)
        dist= Categorical(logits= logits)
        action= dist.sample()
        # print(f"action is:{action}")
        step_log=dist.log_prob(action)
        log_probs.append(step_log)
        observation, reward, terminate, truncate, _ = env.step(action.item())
        rewardlist.append(reward)
        score += reward
        done = terminate or truncate
    total_rewards.append(score)
    discounted_list = []
    g = 0
    for r in reversed(rewardlist):
        g = r+ gamma * g
        discounted_list.insert(0, g)
    total_disc.extend(discounted_list) #appending only the final discounted reward and last step_log
    total_probs.extend(log_probs)
   
    policy_loss= []
    disc_reward_tensor= torch.tensor(total_disc, dtype= torch.float32)
    disc_reward_tensor= (disc_reward_tensor- disc_reward_tensor.mean())/ (disc_reward_tensor.std()+ 1e-8)
    for step_reward, log_prob in zip(disc_reward_tensor, total_probs):
        policy_loss.append(-log_prob*step_reward)

    optimizer.zero_grad()
    loss= torch.stack(policy_loss).sum()
    loss.backward()
    optimizer.step()
    total_disc= []
    total_probs = []

    print(f"episode {i} is done reward is {score}")

env.close()
y_data= np.array(total_rewards)
plt.plot(y_data)
plt.show()
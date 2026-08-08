import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

env= gym.make("CartPole-v1", render_mode= "human")
episodes= 1000
observation, info= env.reset()
print(f"observations are:{observation}")
cart_pos, cart_vel, pole_angle, pole_w= observation

print(f"cart pos is:{cart_pos}")


model= nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16,2)

)
learning_rate= 0.001
optimizer = optim.Adam(model.parameters(), learning_rate)
score=0
done= False
best_score=0
gamma = 0.99
print(f"actions are {env.action_space}")
total_rewards= []
for i in range(0, episodes):
    observation, _ = env.reset()
    done= False
    score= 0
    rewardlist= []
    log_probs= []
    policy_loss= []
    while not done: #agent
        input_tensor= torch.FloatTensor(observation)
        output= model(input_tensor)
        dist= Categorical(logits= output)
        action= dist.sample()
        # print(f"action is:{action}")
        step_log=dist.log_prob(action)
        log_probs.append(step_log)
        observation, reward, terminate, truncate, info2 = env.step(action.item())
        # rewardlist.append(reward)
        score += reward
        rewardlist.append(reward)
        done = terminate or truncate
        # policy_loss.append(-step_log* reward) #torch is for gradient descent and we want ascent. so - sign
    #cant do this cuz all the step wise rewards are +1 so mean =1 only
    # mean=0
    # deviation=0
    # count=0
    # meansquare=0
    # for a in rewardlist:
    #     mean+=a
    #     meansquare+= a**2
    #     count+=1
    # mean=mean/count
    # meansquare= meansquare/count
    # deviation= sqrt(meansquare- mean**2)
    # for step_reward, a in zip(rewardlist, log_probs):
    #     policy_loss.append(-a*((step_reward-mean)/deviation))
    discounted_list= []
    g=0
    for r in reversed(rewardlist):
        g= r+ gamma*g
        discounted_list.insert(0, g)
    for step_reward, a in zip(discounted_list, log_probs):
       policy_loss.append(-a*step_reward)
    
    total_rewards.append(score)
    optimizer.zero_grad()
    loss= torch.stack(policy_loss).sum()
    loss.backward()
    optimizer.step()
    print(f"episode {i} is done reward is {score}")
env.close()
y_data= np.array(total_rewards)
plt.plot(y_data)
plt.show()

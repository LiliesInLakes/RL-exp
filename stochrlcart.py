import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
render_mode= None
env= gym.make("CartPole-v1", render_mode= render_mode)
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
batch_size=5
print(f"actions are {env.action_space}")
total_rewards= []
total_disc= []
total_probs = []
for i in range(0, episodes):
    should_render= ((i+1)%100==0)
    desired_render_mode= "human" if should_render else None
    if desired_render_mode!=render_mode:
        env.close()
        render_mode= desired_render_mode
        env= gym.make("CartPole-v1", render_mode= render_mode)
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
   
    if (i+1) % batch_size == 0:
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
env.close()
y_data= np.array(total_rewards)
plt.plot(y_data)
plt.show()

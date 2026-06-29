import gymnasium as gym
import torch
from torch import nn
env= gym.make("CartPole-v1", render_mode= "human")
episodes= 10
observation, info= env.reset()
print(f"observations are:{observation}")
cart_pos, cart_vel, pole_angle, pole_w= observation

print(f"cart pos is:{cart_pos}")

score=0
done= False

class SeqMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

for i in range(0, episodes):
    env.reset()
    done= False
    score= 0
    while not done:
        action= env.action_space.sample()
        # print(f"action is{action}")
        observation, reward, terminate, truncate, info2= env.step(action)
        # action is done. need to check obs and if term or not and reward!!
        score += reward
        #need to input reward also
        done = terminate or truncate
    print(f"episode {i} is done reward is {score}")
env.close()




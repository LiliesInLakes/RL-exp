import gymnasium as gym
# import pytorch as torch
import numpy as np
import random
env= gym.make("CartPole-v1", render_mode= "human")
episodes= 20
observation, info= env.reset()
print(f"observations are:{observation}")
cart_pos, cart_vel, pole_angle, pole_w= observation

print(f"cart pos is:{cart_pos}")

score=0
done= False
learning_rate= 0.1
#actually lr doesnt have much point here.
weights= [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
print(weights)
best_score=0
print(f"actions are {env.action_space}")
#this will make it do once right??
for i in range(0, episodes):
    env.reset()
    done= False
    score= 0
    rewardlist= []
    noise= [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
    new_weights= [None] * 4
    for j in range(0, 4):
        new_weights[j]= noise[j]+ weights[j]
    while not done: #agent

        dp= np.dot(new_weights, observation)
        action= 0 if dp < 0 else 1 #policy

        # print(f"action is{action}")
        observation, reward, terminate, truncate, info2= env.step(action)
        # action is done. need to check obs and if term or not and reward!!
        # print(f"reward is{reward}")
        # print(f"obs is{observation}")
        rewardlist.append(reward)
        score += reward
        #need to input reward also
        done = terminate or truncate
    if score> best_score:
        best_score=score
        for k in range(0, 4):
            new_weights[k]= learning_rate*(noise[k]+ weights[k])
        print(f"--> New best score found! Saving these weights.\n")
    print(f"episode {i} is done reward is {score}")
env.close()


# Thoughts on rlcart.py

# it is deterministic and rigidly linear. So it cant handle complexities of the enironment and act on them. It has no idea of why the new weights are better
# If i had used a ai model instead, i would get probabilites instead of a hard 0 and 1. that could handle the complexities
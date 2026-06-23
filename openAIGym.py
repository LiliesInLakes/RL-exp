# create gym
# define state 
# make it take an action based on random probabilty
# see the actions
# Next-> add weights instead of random probs






import gymnasium as gym

env= gym.make("CartPole-v1", render_mode= "human")
episodes= 10
observation, info= env.reset()
print(f"observations are:{observation}")
cart_pos, cart_vel, pole_angle, pole_w= observation

print(f"cart pos is:{cart_pos}")

score=0
done= False
#this will make it do once right??
while not done:
    action= env.action_space.sample()
    observation, reward, terminate, truncate, info2= env.step(action)
    # action is done. need to check obs and if term or not and reward!!
    score += reward
    #need to input reward also
    done = terminate or truncate
print(f"episode is done reward is {score}")
env.close()



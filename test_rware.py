import gymnasium as gym
import rware
import time

env = gym.make("rware-tiny-2ag-v2", render_mode="human")

obs, info = env.reset(seed=42)

for step in range(200):
    actions = env.action_space.sample()
    obs, rewards, done, truncated, info = env.step(actions)

    env.render()          # IMPORTANT
    time.sleep(0.1)       # slows it down so you can see

    if done or truncated:
        break

env.close()
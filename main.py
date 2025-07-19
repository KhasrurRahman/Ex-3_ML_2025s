import time
import random
import matplotlib.pyplot as plt
from breakout_env import BreakoutEnv

def plot_env(env):
    plt.clf()
    ax = plt.gca()
    ax.set_xlim(0, env.grid_width)
    ax.set_ylim(0, env.grid_height)
    # Draw bricks
    for bx, by in env.bricks:
        rect = plt.Rectangle((bx, by), env.brick_width, env.brick_height, color='orange')
        ax.add_patch(rect)
    # Draw paddle
    paddle = plt.Rectangle((env.paddle_x, env.paddle_y), env.paddle_width, 1, color='blue')
    ax.add_patch(paddle)
    # Draw ball
    plt.plot(env.ball_x + 0.5, env.ball_y + 0.5, 'ro', markersize=12)
    plt.axis('off')
    plt.pause(0.1)

env = BreakoutEnv(num_bricks=10)
state = env.reset(layout="block")  # You can switch to 'block', 'pyramid', or 'random'

plt.ion()
fig = plt.figure(figsize=(6, 4))

total_reward = 0
for _ in range(200):
    plot_env(env)
    action = random.choice([-1, 0, 1])
    state, reward, done = env.step(action)
    total_reward += reward
    time.sleep(0.2)
    if done:
        print("All bricks cleared! Final Reward:", reward)
        break

print("Total episode reward:", total_reward)

plt.ioff()
plt.show()
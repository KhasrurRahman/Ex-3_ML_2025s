import random, time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from breakout_env import BreakoutEnv

ACTIONS = [-1,0,1]
START_TRAJECTORIES = [(-2,1),(-1,1),(0,1),(1,1),(2,1)]
START_X_OFFSETS = [-2,0,2]

SURVIVAL_STEP_BONUS = 50
SURVIVAL_INTERVAL = 50

def discretize_state(state):
    ball_x, ball_y, ball_vx, ball_vy, paddle_x, bricks = state
    rel_x = ball_x - paddle_x
    return (
        int(ball_x//3),
        int(ball_y//3),
        np.sign(ball_vx),
        np.sign(ball_vy),
        int(rel_x//3),
        1 if len(bricks)>0 else 0
    )

class MonteCarloBreakoutAgent:
    def __init__(self, gamma=0.9, epsilon=0.2, epsilon_min=0.05, decay_rate=0.995, max_steps=3000):
        self.Q = defaultdict(lambda: np.zeros(len(ACTIONS)))
        self.returns = defaultdict(list)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.max_steps = max_steps
        self.q_sizes = []
        self.avg_rewards = []
        self.layout_runtimes = {}

    def policy(self, state):
        if random.random() < self.epsilon or state not in self.Q:
            return random.choice(range(len(ACTIONS)))
        return np.argmax(self.Q[state])

    def generate_episode(self, env: BreakoutEnv, layout="block", start_cfg=None):
        if start_cfg:
            vx, offset = start_cfg
            env.reset(layout=layout, paddle_offset=offset)
            env.ball_vx, env.ball_vy = vx, 1
            env.ball_x = env.grid_width // 2
        else:
            env.reset(layout=layout)

        s = discretize_state(env._get_state())
        done = False
        total_reward = 0
        trajectory = [(env.ball_x, env.ball_y)]
        for step in range(self.max_steps):
            a_idx = self.policy(s)
            s_next, reward, done = env.step(ACTIONS[a_idx])
            reward *= 0.1
            if env.ball_y == env.paddle_y + 1:
                reward += 5
            if len(env.bricks) < 4:
                reward += 20
            if (step+1) % SURVIVAL_INTERVAL == 0:
                reward += SURVIVAL_STEP_BONUS
            s_next = discretize_state(s_next)
            episode_state = s
            s = s_next
            trajectory.append((env.ball_x, env.ball_y))
            total_reward += reward
            yield (episode_state, a_idx, reward), total_reward, trajectory, done
            if done:
                break

    def run_episode_collect(self, env, layout, start_cfg=None):
        episode = []
        total_reward = 0
        traj = []
        for (s,a_idx,r), tr, trajectory, done in self.generate_episode(env,layout,start_cfg):
            episode.append((s,a_idx,r))
            total_reward = tr
            traj = trajectory
            if done:
                break
        return episode, total_reward, traj

    def update(self, episode):
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            state, action_idx, reward = episode[t]
            G = self.gamma * G + reward
            if (state, action_idx) not in visited:
                self.returns[(state, action_idx)].append(G)
                self.Q[state][action_idx] = np.mean(self.returns[(state, action_idx)])
                visited.add((state, action_idx))

    def train_all_start_configs(self, episodes_per_cfg=10, layout_list=["line","block","pyramid"], num_bricks=4):
        env = BreakoutEnv(num_bricks=num_bricks)
        all_start_cfgs = [(vx,offset) for vx,_ in START_TRAJECTORIES for offset in START_X_OFFSETS]
        for layout in layout_list:
            start_time = time.time()
            for start_cfg in all_start_cfgs:
                for ep in range(episodes_per_cfg):
                    episode, total_reward, _ = self.run_episode_collect(env, layout, start_cfg)
                    self.update(episode)
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.decay_rate
                    self.q_sizes.append(len(self.Q))
                    self.avg_rewards.append(total_reward)
            runtime = time.time() - start_time
            self.layout_runtimes[layout] = runtime
            print(f"Runtime for layout {layout} over all 15 start configs: {runtime:.2f}s")
        self.plot_metrics()

    def plot_metrics(self):
        plt.figure(figsize=(10,3))
        plt.subplot(1,2,1)
        plt.plot(self.q_sizes)
        plt.title("Q-size growth")
        plt.subplot(1,2,2)
        plt.plot(self.avg_rewards)
        plt.title("Episode rewards")
        plt.show()
        if self.layout_runtimes:
            plt.bar(self.layout_runtimes.keys(), self.layout_runtimes.values())
            plt.title("Runtime per Layout")
            plt.ylabel("seconds")
            plt.show()

    def evaluate_policy(self, layout="block", runs=1, plot_trajectory=True, print_layout=True, start_cfg=None):
        env = BreakoutEnv()
        for run in range(runs):
            if start_cfg:
                vx, offset = start_cfg
                env.reset(layout=layout, paddle_offset=offset)
                env.ball_vx, env.ball_vy = vx, 1
                env.ball_x = env.grid_width // 2
            else:
                env.reset(layout=layout)

            s = discretize_state(env._get_state())
            total_reward = 0
            trajectory = [(env.ball_x, env.ball_y)]

            initial_paddle_x = env.paddle_x
            initial_bricks = list(env.bricks)
            start_ball_x, start_ball_y = env.ball_x, env.ball_y
            done = False
            for step in range(self.max_steps):
                a_idx = np.argmax(self.Q[s]) if s in self.Q else random.choice(range(len(ACTIONS)))
                s_next, reward, done = env.step(ACTIONS[a_idx])
                if done:
                    print("WIN")
                reward *= 0.1
                if env.ball_y == env.paddle_y + 2:
                    reward += 5
                if len(env.bricks) < 4:
                    reward += 20
                if (step + 1) % SURVIVAL_INTERVAL == 0:
                    reward += SURVIVAL_STEP_BONUS
                s = discretize_state(s_next)
                total_reward += reward
                trajectory.append((env.ball_x, env.ball_y))
                if done:
                    break
            print(f"Run {run + 1} reward: {total_reward}")

            if plot_trajectory:
                fig, ax = plt.subplots()

                for bx, by in initial_bricks:
                    rect = plt.Rectangle((bx, by), env.brick_width, env.brick_height, color='orange', alpha=0.5)
                    ax.add_patch(rect)

                paddle_rect = plt.Rectangle((initial_paddle_x, env.paddle_y), env.paddle_width, 1, color='blue',
                                            alpha=0.5)
                ax.add_patch(paddle_rect)

                xs, ys = zip(*trajectory)
                ax.plot(xs, ys, marker='o', color='red', label='Trajectory')

                ax.scatter(xs[0], ys[0], color='green', s=80, label='Start')
                ax.scatter(xs[-1], ys[-1], color='red', s=80, label='End')

                win_status = "Victory!" if done else "Game Over"
                ax.set_title(
                    f"{win_status} | Reward: {total_reward:.1f}\nLayout: {layout}")
                ax.set_xlim(0, env.grid_width)
                ax.set_ylim(0, env.grid_height)
                ax.legend()
                plt.gca().invert_yaxis()
                plt.show()

if __name__ == "__main__":
    agent = MonteCarloBreakoutAgent()
    agent.train_all_start_configs(episodes_per_cfg=50, layout_list=["line", "pyramid", "block"], num_bricks=4)

    all_start_cfgs = [(vx, offset) for vx, _ in START_TRAJECTORIES for offset in START_X_OFFSETS]

    for layout in ["line", "block", "pyramid"]:
        #print(f"\nEvaluating all start configs on layout: {layout}")
        for i, start_cfg in enumerate(all_start_cfgs):
            #print(f"\n--- Start config {i + 1}: vx={start_cfg[0]}, paddle_offset={start_cfg[1]} ---")
            agent.evaluate_policy(layout=layout, runs=1, plot_trajectory=True, start_cfg=start_cfg)


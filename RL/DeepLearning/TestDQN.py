#%%
# experiment.py

from DQN import DQNAgent, ReplayBuffer, Net
import gym
import numpy as np

def train_dqn(agent, num_episodes=500):
    env = gym.make('CartPole-v1')

    sync_freq = 10  # ターゲットネットワークの同期頻度
    rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward

        agent.update_epsilon()
        rewards.append(total_reward)

        if episode % sync_freq == 0:
            agent.sync_qnet()

        print(f"Episode {episode}, Total Reward: {total_reward}")

    env.close()

    return rewards

if __name__ == "__main__":
    agent = DQNAgent()
    trained_rewards = train_dqn(agent)
    # 学習の結果をどこかに保存する場合は、ここに保存処理を追加する

# %%

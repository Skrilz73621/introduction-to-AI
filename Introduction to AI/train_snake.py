# train.py
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from reinforcment_game import SnakeGame, DQN  # импортируем игру и модель из другого файла
import pickle  # для сохранения модели

def train():
    episodes = 1000
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.05
    lr = 0.001
    batch_size = 64
    memory = deque(maxlen=5000)

    env = SnakeGame()
    model = DQN(6, 4)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    q_values = model(torch.tensor(state))
                    action = torch.argmax(q_values).item()

            next_state, reward, done = env.step(action)
            total_reward += reward

            memory.append((state, action, reward, next_state, done))
            state = next_state

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards_, next_states, dones = zip(*batch)

                states = torch.tensor(states)
                actions = torch.tensor(actions)
                rewards_ = torch.tensor(rewards_, dtype=torch.float32)
                next_states = torch.tensor(next_states)
                dones = torch.tensor(dones, dtype=torch.bool)

                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q_values = model(next_states).max(1)[0]
                targets = rewards_ + gamma * next_q_values * (~dones)

                loss = loss_fn(q_values, targets.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards.append(total_reward)
        print(f"Episode {ep+1}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.grid()
    plt.show()

    # Сохраняем модель
    torch.save(model.state_dict(), "dqn_snake.pth")
    print("✅ Модель сохранена как dqn_snake.pth")


if __name__ == "__main__":
    train()

import time  # добавляем для замедления
import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# === Параметры среды ===
WIDTH, HEIGHT = 800, 800
BLOCK_SIZE = 80
GRID_WIDTH = WIDTH // BLOCK_SIZE
GRID_HEIGHT = HEIGHT // BLOCK_SIZE

# === Действия: 0 - Влево, 1 - Вправо, 2 - Вверх, 3 - Вниз ===
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# === Модель DQN ===
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# === Класс змейки ===
class SnakeGame:
    def __init__(self):
        self.obstacles = [
        (3, 1), (4, 3), (5, 3),
        (6, 6), (6, 7), (6, 8),
        (2, 10), (3, 10), (4, 10)
    ]

    def reset(self):
        self.snake = [(5, 5)]
        self.direction = (1, 0)
        self.spawn_food()
        self.score = 0
        self.frame = 0
        self.steps_since_last_food = 0
        return self.get_state()

    def spawn_food(self):
        while True:
            self.food = (
                random.randint(0, GRID_WIDTH - 1),
                random.randint(0, GRID_HEIGHT - 1)
            )
            if self.food not in self.snake and self.food not in self.obstacles:
                break


    def spawn_obstacles(self):
        self.obstacles = []
        for _ in range(5):  # 👈 количество препятствий
            while True:
                obs = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
                if obs not in self.snake and obs != self.food and obs not in self.obstacles:
                    self.obstacles.append(obs)
                    break

    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        return np.array([
            head_x / GRID_WIDTH,
            head_y / GRID_HEIGHT,
            food_x / GRID_WIDTH,
            food_y / GRID_HEIGHT,
            self.direction[0],
            self.direction[1]
        ], dtype=np.float32)

    def step(self, action_idx):
        action = ACTIONS[action_idx]
        self.direction = action
        head = (self.snake[0][0] + action[0], self.snake[0][1] + action[1])
        
        prev_distance = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
        self.snake.insert(0, head)

        reward = -0.01
        done = False

        new_distance = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

        if head == self.food:
            self.score += 1
            reward = 1
            self.spawn_food()
            self.steps_since_last_food = 0
        else:
            self.snake.pop()
            if new_distance < prev_distance:
                reward += 0.01
            else:
                reward -= 0.02
        
    
    
        # Условия поражения
        if (
            head[0] < 0 or head[0] >= GRID_WIDTH or
            head[1] < 0 or head[1] >= GRID_HEIGHT or
            head in self.snake[1:] or
            head in self.obstacles  # 👈 столкновение с препятствием
        ):
            done = True
            reward = -1

        self.steps_since_last_food += 1
        if self.steps_since_last_food > 100:
            done = True
            reward = -1

        self.frame += 1
        return self.get_state(), reward, done



# === Функция тренировки DQN ===
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

    # === Визуализация прогресса обучения ===
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Progress of DQN Agent in Snake')
    plt.grid()
    plt.show()

    # После тренировки — запускаем игру
    return model

# === Визуализация игры ===
def play(model):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    env = SnakeGame()
    state = env.reset()

    running = True
    while running:
        clock.tick(10)  # 👈 скорость игры: 10 кадров в секунду

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Агент выбирает лучшее действие
        with torch.no_grad():
            q_values = model(torch.tensor(state))
            action = torch.argmax(q_values).item()

        next_state, reward, done = env.step(action)
        state = next_state

        if done:
            state = env.reset()

               # === Отрисовка ===
        screen.fill((0, 0, 0))  # Чёрный фон

        # Препятствия
        # Препятствия (тёмно-серые блоки)
        for obs in env.obstacles:
            obs_rect = pygame.Rect(obs[0]*BLOCK_SIZE, obs[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(screen, (80, 80, 80), obs_rect)


        # Еда
        food_rect = pygame.Rect(env.food[0]*BLOCK_SIZE, env.food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(screen, (255, 0, 0), food_rect)

        # Тело змейки
        for part in env.snake:
            snake_rect = pygame.Rect(part[0]*BLOCK_SIZE, part[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(screen, (0, 255, 0), snake_rect)


        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    trained_model = train()
    play(trained_model)


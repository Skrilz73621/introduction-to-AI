# play.py
import pygame
import torch
from reinforcment_game import SnakeGame, DQN

# Исходные размеры
WIDTH, HEIGHT = 800, 800
BLOCK_SIZE = 80

def play():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    # Загрузка спрайтов
    head_img_original = pygame.image.load("assets/snakehead.png").convert_alpha()
    body_img = pygame.image.load("assets/snakebody.png").convert_alpha()
    food_img = pygame.image.load("assets/snakefood.png").convert_alpha()

    # Масштабируем под BLOCK_SIZE
    head_img_original = pygame.transform.scale(head_img_original, (BLOCK_SIZE, BLOCK_SIZE))
    body_img = pygame.transform.scale(body_img, (BLOCK_SIZE, BLOCK_SIZE))
    food_img = pygame.transform.scale(food_img, (BLOCK_SIZE, BLOCK_SIZE))

    env = SnakeGame()
    state = env.reset()

    model = DQN(6, 4)
    model.load_state_dict(torch.load("dqn_snake.pth"))  # загружаем модель
    model.eval()

    running = True
    while running:
        clock.tick(10)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        with torch.no_grad():
            q_values = model(torch.tensor(state))
            action = torch.argmax(q_values).item()

        next_state, reward, done = env.step(action)
        state = next_state

        if done:
            state = env.reset()

        # Отрисовка
        screen.fill((0, 0, 0))  # черный фон

        # Еда
        screen.blit(food_img, (env.food[0] * BLOCK_SIZE, env.food[1] * BLOCK_SIZE))
        
        # Отрисовка препятствий (тёмно-серые блоки)
        for obs in env.obstacles:
            obs_rect = pygame.Rect(obs[0]*BLOCK_SIZE, obs[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(screen, (80, 80, 80), obs_rect)


        # Змейка
        for idx, part in enumerate(env.snake):
            x, y = part[0] * BLOCK_SIZE, part[1] * BLOCK_SIZE
            if idx == 0:
                # Поворачиваем голову в нужном направлении
                dx, dy = env.direction
                angle = 0
                if dx == 1:
                    angle = 90
                elif dx == -1:
                    angle = -90
                elif dy == 1:
                    angle = 0
                elif dy == -1:
                    angle = 180
                head_img = pygame.transform.rotate(head_img_original, angle)
                screen.blit(head_img, (x, y))
            else:
                screen.blit(body_img, (x, y))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    play()

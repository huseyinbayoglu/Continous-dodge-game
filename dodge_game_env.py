import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
from gymnasium.envs.registration import register
import numpy as np
import pygame


class DodgeGameEnv(gym.Env):
    def __init__(self, width=500, height=500, number_of_balls=0):
        super().__init__()
        
        self.width = width
        self.height = height
        self.number_of_balls = number_of_balls
        self.agent_size = 20  # Kare ajanın boyutu
        self.ball_size = 20  # Kare topların boyutu

        self.rewards = {
            "collision": -6,
            "closer": .5,
            "farer":-.6,
            "achived":2
        }
        
        # Gymnasium action & observation spaces
        self.action_space = Discrete(5)  # 4 yönlü hareket (Up, Down, Left, Right)
        self.observation_space = Box(0, 1, shape=(2,), dtype=np.float32)
        """self.observation_space = Dict({
            "agent": Box(0, max(self.width, self.height), shape=(2,), dtype=np.int32),
            "target": Box(0, max(self.width, self.height), shape=(2,), dtype=np.int32),
            "balls": Box(0, max(self.width, self.height), shape=(number_of_balls, 2), dtype=np.int32),
        })"""
        
        # Hareket yönleri
        self._action_to_direction = {
            0: np.array([0, -10]),  # Sol (Left)
            1: np.array([0, 10]),   # Sağ (Right)
            2: np.array([-10, 0]),  # Yukarı (Up)
            3: np.array([10, 0]),   # Aşağı (Down)
            4: np.array([0,0])      # Nothing
        }
        
        # Pygame başlat
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Ajan ve hedef konumları
        self.agent_rect = pygame.Rect(
            self.np_random.integers(0, self.width - self.agent_size),
            self.np_random.integers(0, self.height - self.agent_size),
            self.agent_size, self.agent_size
        )
        
        # Hedefin ajan üzerinde spawn olmamasını sağla
        while True:
            self.target_rect = pygame.Rect(
                self.np_random.integers(0, self.width - self.agent_size),
                self.np_random.integers(0, self.height - self.agent_size),
                self.agent_size, self.agent_size
            )
            if not self.target_rect.colliderect(self.agent_rect):
                break
        
        # Topların başlangıç konumları ve hareket yönleri
        self.ball_rects = []
        for _ in range(self.number_of_balls):
            while True:
                ball_rect = pygame.Rect(
                    self.np_random.integers(0, self.width - self.ball_size),
                    self.np_random.integers(0, self.height - self.ball_size),
                    self.ball_size, self.ball_size
                )
                if not ball_rect.colliderect(self.agent_rect):
                    self.ball_rects.append(ball_rect)
                    break
        
        self.ball_directions = self.np_random.choice([-1, 1], size=(self.number_of_balls, 2)) * 5  # -5 veya 5 hız
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        # Normalize değerler için genişlik ve yükseklik
        norm_width = self.width
        norm_height = self.height

        """target_distance_x = self.target_rect.centerx - self.agent_rect.centerx
        target_distance_x = (target_distance_x // 10) -1 if target_distance_x >0 else (target_distance_x // 10) + 1
        target_distance_y = self.target_rect.centery - self.agent_rect.centery
        target_distance_y = (target_distance_y // 10) -1 if target_distance_y >0 else (target_distance_y // 10) + 1
        target_distance_x /= (self.width // 10)
        target_distance_y /= (self.height // 10)"""
        # Ajanın hedefe olan normalize konumu (x farkı, y farkı)
        if (self.target_rect.centerx - self.agent_rect.centerx) < 0:
            agent_target_dx = -1
        if (self.target_rect.centerx - self.agent_rect.centerx) > 0:
            agent_target_dx = 1
        if abs((self.target_rect.centerx - self.agent_rect.centerx)) < self.agent_size:
            agent_target_dx = 0

        if (self.target_rect.centery - self.agent_rect.centery) < 0:
            agent_target_dy = -1
        if (self.target_rect.centery - self.agent_rect.centery) > 0:
            agent_target_dy = 1
        if abs((self.target_rect.centery - self.agent_rect.centery)) < self.agent_size:
            agent_target_dy = 0


        # Top bilgileri (Ajan-top mesafesi + Topun yönü)
        balls_data = []
        for i, ball in enumerate(self.ball_rects):
            ball_dx = (ball.centerx - self.agent_rect.centerx) / norm_width
            ball_dy = (ball.centery - self.agent_rect.centery) / norm_height

            # Top yönleri (-1 veya 1 olarak)
            ball_dir_x = self.ball_directions[i][0] / abs(self.ball_directions[i][0])
            ball_dir_y = self.ball_directions[i][1] / abs(self.ball_directions[i][1])

            balls_data.extend([ball_dx, ball_dy, ball_dir_x, ball_dir_y])

        # Gözlem vektörünü oluştur
        return np.array([agent_target_dx, agent_target_dy] + balls_data, dtype=np.float32)

    def step(self, action):
        first_distance = abs(self.target_rect.left - self.agent_rect.left) + abs(self.target_rect.top - self.agent_rect.top)
        # Ajanı hareket ettir
        movement = self._action_to_direction[int(action)]
        self.agent_rect.move_ip(movement[1], movement[0])  # (dx, dy)

        second_distance = abs(self.target_rect.left - self.agent_rect.left) + abs(self.target_rect.top - self.agent_rect.top)

        # Duvar çarpışmasını engelle (ajan)
        self.agent_rect.clamp_ip(pygame.Rect(0, 0, self.width, self.height))
        
        # Topları hareket ettir ve duvar çarpışmalarını kontrol et
        for i, ball in enumerate(self.ball_rects):
            ball.move_ip(self.ball_directions[i][0], self.ball_directions[i][1])
            
            if ball.left <= 0 or ball.right >= self.width:
                self.ball_directions[i][0] *= -1  # X yönünü ters çevir
            if ball.top <= 0 or ball.bottom >= self.height:
                self.ball_directions[i][1] *= -1  # Y yönünü ters çevir
        
        terminated = False
        truncated = False  # Zaman sınırı varsa True yapılabilir
        reward = self.rewards["closer"] if second_distance < first_distance else self.rewards["farer"]


        if self.agent_rect.colliderect(self.target_rect):
            reward = self.rewards["achived"]
            # self._place_target()  
            terminated = True  
            return self._get_obs(), reward, terminated, truncated, {}

        for ball in self.ball_rects:
            if self.agent_rect.colliderect(ball):
                reward = self.rewards["collision"]
                terminated = True
                break
        

        return self._get_obs(), reward, terminated, truncated, {}
    
    def render(self):
        RED = (255,0,0)
        GREEN = (0,255,0)
        YELLOW = (255,0,255)
        self.screen.fill((0, 0, 0))  # Arka plan siyah

        # Ajanı ve hedefi kare olarak çiz
        pygame.draw.rect(self.screen, GREEN, self.agent_rect)  # Yeşil ajan
        pygame.draw.rect(self.screen, YELLOW, self.target_rect)  # Kırmızı hedef

        # Topları yuvarlak olarak çiz
        for ball in self.ball_rects:
            pygame.draw.circle(self.screen, RED, ball.center, self.ball_size // 2)
        
        pygame.event.pump()  # Olayları güncelle
        pygame.display.flip()
        self.clock.tick(45)  # FPS sınırı

    def _place_target(self):
        self.target_rect = pygame.Rect(
            self.np_random.integers(0, self.width - self.agent_size),
            self.np_random.integers(0, self.height - self.agent_size),
            self.agent_size, self.agent_size
        )

    def close(self):
        pygame.quit()

register(
    id="DodgeGame-v0",
    entry_point=__name__ + ":DodgeGameEnv",
)
"""
if __name__ == "__main__":
    env = DodgeGameEnv(number_of_balls=13, width=500, height=500)
    obs, _ = env.reset()
    done = False
    over = False
    action = 4  # Başlangıçta hareketsiz

    while not over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                over = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    action = 2  # Yukarı
                elif event.key == pygame.K_s:
                    action = 3  # Aşağı
                elif event.key == pygame.K_a:
                    action = 0  # Sol
                elif event.key == pygame.K_d:
                    action = 1  # Sağ
                
                obs, reward, done, _, _ = env.step(action)
                obs_dict = {
                    "relative_target_position": obs[:2].tolist(),
                    "relative_ball_positions": obs[2:].tolist()
                }
                print(obs_dict)
                env.render()
                if done:
                    obs, _ = env.reset()
    
    env.close()

"""


"""if __name__ == "__main__":
    def make_env(number_of_balls=14, width=16000, height=16000):
        return DodgeGameEnv(number_of_balls=number_of_balls, width=width, height=height)

    n_env = 10
    envs = [make_env() for _ in range(n_env)]
    obss = [env.reset()[0] for env in envs]
    dones = [False] * n_env
    rewards = [None] * n_env
    lengths = [0] * n_env
    
    import time 
    
    over = False
    st = time.time()
    
    while not over:
        over = all(dones)  # Eğer tüm environment'lar done olduysa, döngüyü bitir
        
        for idx, env in enumerate(envs):
            if not dones[idx]:  # Sadece bitmemiş environment'lar için adım at
                action = np.random.randint(0,4)
                obss[idx], rewards[idx], dones[idx], _, _ = env.step(action)
                lengths[idx] += 1
    
    elapsed_time = time.time() - st
    total_steps = sum(lengths)
    fps = round(total_steps / elapsed_time) if elapsed_time > 0 else 0
    
    print(f"Geçen süre: {elapsed_time:.2f} saniye\nToplam Adım: {total_steps}\nFPS: {fps}")
"""

if __name__ == "__main__":
    env = DodgeGameEnv()
    obs, _ = env.reset()
    print(obs.shape)
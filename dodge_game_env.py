import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from gymnasium.envs.registration import register
import numpy as np
import pygame
import pickle
import os 

# TODO get obs metodunda kategorize verileri onehot encoding ile ver✅

# TODO make a function that save the entire env state and make another function (_set state) that
# load the saved state ✅

class DodgeGameEnv(gym.Env):
    def __init__(self, width=500, height=500, number_of_balls=13):
        super().__init__()
        
        self.width = width
        self.height = height
        self.number_of_balls = number_of_balls
        self.agent_size = 20  # Kare ajanın boyutu
        self.ball_size = 20  # Kare topların boyutu

        self.rewards = {
            "collision": -5,
            "closer": .4,
            "farer":-.7,
            "achived":5
        }
        
        # Gymnasium action & observation spaces
        self.action_space = Discrete(5)  # 4 yönlü hareket (Up, Down, Left, Right)
        self.observation_space = Box(0, 1, shape=(334,), dtype=np.float32)
        
        
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
        def get_direction_and_distance(x1, y1, x2, y2, norm_factor_x, norm_factor_y, agent_size):
            """Hedef veya top için yön (-1, 0, 1) ve normalize mesafe hesaplar."""
            dx = x2 - x1
            dy = y2 - y1
            
            # Yön hesaplama
            direction_x = -1 if dx < 0 else (1 if dx > 0 else 0)
            direction_y = -1 if dy < 0 else (1 if dy > 0 else 0)

            # Eğer ajan hedefe/topa yeterince yakınsa, yön sıfırlanır
            if abs(dx) < agent_size:
                direction_x = 0
            if abs(dy) < agent_size:
                direction_y = 0

            # Normalize mesafe hesaplama
            norm_dx = abs(dx) // 10  # number of step
            norm_dy = abs(dy) // 10  # number of step

            distance_encoding_x = [0]*10
            distance_encoding_y = [0]*10

            if 0<norm_dx < 9:
                distance_encoding_x[norm_dx] = 1
            elif norm_dx == 0:
                distance_encoding_x = [0]*10
            else:
                distance_encoding_x[9] = 1  # The last index
            
            if 0<norm_dy < 9:
                distance_encoding_y[norm_dy] = 1
            elif norm_dy == 0:
                distance_encoding_y = [0]*10
            else:
                distance_encoding_y[9] = 1  # The last index
            

            return direction_x, direction_y, distance_encoding_x, distance_encoding_y


        norm_width = self.width // 10
        norm_height = self.height // 10

        # Ajan ve hedef arasındaki yön ve mesafe
        agent_target_dx, agent_target_dy, norm_dx, norm_dy = get_direction_and_distance(
            self.agent_rect.centerx, self.agent_rect.centery,
            self.target_rect.centerx, self.target_rect.centery,
            norm_width, norm_height, self.agent_size
        )

        # Top bilgileri
        balls_data = []
        for i,ball in enumerate(self.ball_rects):
            agent_ball_dx, agent_ball_dy, cate_dx_ball, cate_dy_ball = get_direction_and_distance(
                self.agent_rect.centerx, self.agent_rect.centery,
                ball.centerx, ball.centery,
                norm_width, norm_height, self.agent_size
            )
            # Topun hız yönü (velocity_x, velocity_y)
            velocity_x, velocity_y = self.ball_directions[i]  # Topların hız vektörleri
            velocity_x /= 5
            velocity_y /= 5

            # Tüm bilgileri ekleyelim
            balls_data.append([agent_ball_dx, agent_ball_dy, cate_dx_ball, cate_dy_ball, velocity_x, velocity_y])

        # Gözlem vektörü oluşturma
        flattened_balls_data = []
        for item in balls_data:
            flattened_balls_data.append([
                item[0],  # direction_x
                item[1],  # direction_y
                *item[2],  # Açılmış distance_encoding_x
                *item[3],  # Açılmış distance_encoding_y
                item[4],  # velocity_x
                item[5]   # velocity_y
            ])
        balls_data = np.array(flattened_balls_data).flatten()
        state = np.concatenate([[agent_target_dx, agent_target_dy, *norm_dx, *norm_dy], balls_data])

        return state

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

    def save_state(self, name:str = "state"):
        if not os.path.exists("states"):
            os.makedirs("states")
        info = {
            "player_coordinates": (self.agent_rect.left, self.agent_rect.top),
            "target_coordinates": (self.target_rect.left, self.target_rect.top),
            "width": self.width,
            "height": self.height,
            "balls": [
                (ball.left, ball.top, self.ball_directions[idx][0], self.ball_directions[idx][1])
                for idx, ball in enumerate(self.ball_rects)
            ],
        }

        file_path = os.path.join("states", f"{name}.pkl")
        with open(file_path, "wb") as dosya:
            pickle.dump(info, dosya)

        return info
    
    def _set_state(self, state=None):
        """Dosya yolunu veya doğrudan state dictionary'sini alarak oyun durumunu yükler."""
        if isinstance(state, str):  # Eğer state bir dosya yoluysa
            file_path = os.path.join("states", f"{state}.pkl")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} file cannot be found.")
            with open(file_path, "rb") as file:
                state = pickle.load(file)
        elif not isinstance(state, dict):
            raise ValueError("Invalid state input: Expected file path or dictionary.")

        # Yüklenen state’i uygula
        self.agent_rect.left, self.agent_rect.top = state["player_coordinates"]
        self.target_rect.left, self.target_rect.top = state["target_coordinates"]
        self.width = state["width"]
        self.height = state["height"]

        self.ball_rects = []
        self.ball_directions = []
        for x, y, dx, dy in state["balls"]:
            self.ball_rects.append(pygame.Rect(x, y, 20, 20))  # Top boyutunu uygun şekilde belirle
            self.ball_directions.append((dx, dy))

        return state


register(
    id="DodgeGame-v0",
    entry_point=__name__ + ":DodgeGameEnv",
)

"""
if __name__ == "__main__":
    env = DodgeGameEnv(number_of_balls=0, width=500, height=500)
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
                
                print(f"hamle öncesi observation: {env._get_obs()}")
                obs, reward, done, _, _ = env.step(action)
                print(f"hamle sonrası observation:{obs}")
                env.render()
                if done:
                    obs, _ = env.reset()
                pygame.time.delay(1500)
                    
    
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
    print(obs.shape,"\n",obs,"\n",env.target_rect,env.agent_rect)
    env.save_state("deneme1")
    env.reset()
    env._set_state("deneme1")

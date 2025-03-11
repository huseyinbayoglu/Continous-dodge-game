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

# TODO get_obs method is not good. It gives not exact information. Fix it

class DodgeGameEnv(gym.Env):
    def __init__(self, width=500, height=500, number_of_balls=3):
        super().__init__()
        
        self.width = width
        self.height = height
        self.number_of_balls = number_of_balls
        self.agent_size = 20  # Kare ajanın boyutu
        self.ball_size = 20  # Kare topların boyutu
        self.agent_speed = 10
        self.ball_speed = 5

        self.rewards = {
            "collision": -6,
            "closer": .3,
            "farer":-.5,
            "achived":2
        }
        
        # Gymnasium action & observation spaces
        self.action_space = Discrete(5)  # 4 yönlü hareket (Up, Down, Left, Right)
        # self.observation_space = Box(0, 1, shape=(334,), dtype=np.float32)
        # self.observation_space = Box(0, 1, shape=(4 + self.number_of_balls * 6,), dtype=np.float32)
        self.observation_space = Box(0, 1, shape=(4+ 6*self.number_of_balls,), dtype=np.float32)
        
        # Hareket yönleri
        self._action_to_direction = {
            0: np.array([0, -self.agent_speed]),  # Sol (Left)
            1: np.array([0, self.agent_speed]),   # Sağ (Right)
            2: np.array([-self.agent_speed, 0]),  # Yukarı (Up)
            3: np.array([self.agent_speed, 0]),   # Aşağı (Down)
            4: np.array([0,0])      # Nothing
        }
        
        # Pygame başlat
        pygame.init()
        self.reset()
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
            if not self._rect_collision_or_touch(self.target_rect,self.agent_rect):
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
                if not self._rect_collision_or_touch(ball_rect,self.agent_rect):
                    self.ball_rects.append(ball_rect)
                    break
        
        self.ball_directions = self.np_random.choice([-1, 1], size=(self.number_of_balls, 2)) * self.ball_speed 
        
        return self._get_obs(), {}

    def _get_obs2(self):  # Categorical
        def get_direction_and_distance(x1, y1, x2, y2,agent_size):
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
            
            distancex = dx / self.width
            distancey = dy / self.height
            

            return direction_x, direction_y, distancex, distancey



        # Ajan ve hedef arasındaki yön ve mesafe
        agent_target_dx, agent_target_dy, distancex,distancey = get_direction_and_distance(
            self.agent_rect.centerx, self.agent_rect.centery,
            self.target_rect.centerx, self.target_rect.centery,self.agent_size
        )



        return np.array([agent_target_dx,agent_target_dy,distancex,distancey])

    def _get_obs(self):
        def get_distance_and_direction(rect1, rect2):
            """
            rect1: Reference rectangle (agent)
            rect2: Target rectangle (target or ball)
            Returns:
            - shortest x distance, shortest y distance
            - relative direction in x, relative direction in y
            """
            distance_x = max(0, (rect2.left - rect1.right) / self.width) if rect1.centerx < rect2.centerx else max(0, (rect1.left - rect2.right) / self.width)
            direction_x = 1 if rect1.centerx < rect2.centerx else -1 
            if distance_x == 0:
                direction_x = 0
            distance_y = max(0, (rect2.top - rect1.bottom) / self.height) if rect1.top < rect2.top else max(0, (rect1.top - rect2.bottom) / self.height)
            direction_y = -1 if rect1.top < rect2.top else 1 
            if distance_y == 0:
                direction_y = 0
            return [distance_x, distance_y], [direction_x, direction_y]
        
        
        # Compute target features
        target_distances, target_directions = get_distance_and_direction(self.agent_rect, self.target_rect)
        
         # Compute ball features
        ball_features = []
        for i, ball in enumerate(self.ball_rects):
            ball_distances, ball_directions = get_distance_and_direction(self.agent_rect, ball)
            ball_velocity_direction = [
                self.ball_directions[i][0] / self.ball_speed,  # -1 or 1
                self.ball_directions[i][1] / self.ball_speed   # -1 or 1
            ]
            
            ball_features.extend(ball_distances)
            ball_features.extend(ball_directions)
            ball_features.extend(ball_velocity_direction)
        
         # Final observation
        final_observation = np.concatenate([target_distances, target_directions, ball_features])
        return np.copy(final_observation.astype(np.float32))

    def step(self, action):
        before_action_state = {
            "player_coordinates": (self.agent_rect.left, self.agent_rect.top),
            "target_coordinates": (self.target_rect.left, self.target_rect.top),
            "width": self.width,
            "height": self.height,
            "balls": [
                (ball.left, ball.top, self.ball_directions[idx][0], self.ball_directions[idx][1])
                for idx, ball in enumerate(self.ball_rects)
            ],
        }


        distancex = 0
        if self.agent_rect.centerx < self.target_rect.centerx: # agent is left of the target
            distancex = max(0,(self.target_rect.left - self.agent_rect.right) / self.width)
        else:
            distancex = max(0,(self.agent_rect.left - self.target_rect.right) / self.width)
        
        distancey = 0
        if self.agent_rect.top < self.target_rect.top: # agent is up of the target
            distancey = max(0,(self.target_rect.top - self.agent_rect.bottom) / self.height)
        else:
            distancey = max(0,(self.agent_rect.top - self.target_rect.bottom) / self.height)

        first_distance = distancex + distancey


        # Ajanı hareket ettir
        movement = self._action_to_direction[int(action)]
        self.agent_rect.move_ip(movement[1], movement[0])  # (dx, dy)


        distancex = 0
        if self.agent_rect.centerx < self.target_rect.centerx: # agent is left of the target
            distancex = max(0,(self.target_rect.left - self.agent_rect.right) / self.width)
        else:
            distancex = max(0,(self.agent_rect.left - self.target_rect.right) / self.width)
        
        distancey = 0
        if self.agent_rect.top < self.target_rect.top: # agent is up of the target
            distancey = max(0,(self.target_rect.top - self.agent_rect.bottom) / self.height)
        else:
            distancey = max(0,(self.agent_rect.top - self.target_rect.bottom) / self.height)
        second_distance = distancex + distancey
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
        # print(f"first distance: {first_distance} second distance:{second_distance} difference:{second_distance-first_distance}")
        reward = self.rewards["closer"] if second_distance < first_distance else self.rewards["farer"]

        if self._rect_collision_or_touch(self.agent_rect,self.target_rect):
            reward = self.rewards["achived"]
            # self._place_target()  
            terminated = True  
            return self._get_obs(), reward, terminated, truncated, {"before_action_state":before_action_state}

        for ball in self.ball_rects:
            if self._rect_collision_or_touch(self.agent_rect,ball):
                reward = self.rewards["collision"]
                terminated = True
                break
        

        return self._get_obs(), reward, terminated, truncated, {"before_action_state":before_action_state}
    
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

    def save_state(self, name:str = "state", state_ = None):
        if not os.path.exists("states"):
            os.makedirs("states")
        if state_ == None:
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
        else:
            info = state_

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

    def _rect_collision_or_touch(self,rect1, rect2):
        # Pygame.recet.colliderect doesn't return True if rects intersect in boundary
        # Normal çarpışmayı kontrol et
        if rect1.colliderect(rect2):
            return True
        
        # Kenardan temas durumlarını kontrol et
        if (rect1.right == rect2.left or rect1.left == rect2.right) and (rect1.top <= rect2.bottom and rect1.bottom >= rect2.top):
            return True
        if (rect1.bottom == rect2.top or rect1.top == rect2.bottom) and (rect1.left <= rect2.right and rect1.right >= rect2.left):
            return True

        return False


register(
    id="DodgeGame-v0",
    entry_point=__name__ + ":DodgeGameEnv",
)


"""if __name__ == "__main__":
    env = DodgeGameEnv(number_of_balls=1, width=500, height=500)
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
                
                print("*"*30)
                print(f"hamle öncesi observation: {env._get_obs()}")
                obs, reward, done, _, _ = env.step(action)
                print(f"hamle sonrası observation:{obs},reward:{reward},done:{done}",end="\n"*3)
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
    env = DodgeGameEnv(number_of_balls=0)
    obs, _ = env.reset()
    print("Without ball\nobs shape: ",obs.shape,"\nobs: ",obs,"\n")
    env = DodgeGameEnv(number_of_balls=1)
    obs, _ = env.reset()
    print("With 1 ball\nobs shape: ",obs.shape,"\nobs: ",obs,"\n")
    env = DodgeGameEnv(number_of_balls=2)
    obs, _ = env.reset()
    print("With 2 ball\nobs shape: ",obs.shape,"\nobs: ",obs,"\n")

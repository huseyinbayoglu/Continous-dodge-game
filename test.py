import gymnasium as gym
from DQN import DQN
import numpy as np
import dodge_game_env  # DodgeGameEnv'nin doğru olduğundan emin ol
import pygame

# TODO Make a test function to test multiple environment and visualize the results



env_name = "DodgeGame-v0"  # Kendi ortamının adını buraya gir
agent = DQN(env_name,n_env=500,train_frequency=5,maxlen=100_000,update_target_frequency=30,
            batch_size=64,gamma=.95, epsilon_decay=.995)
agent.load_model("deneme_son2.keras")
print(agent.main_model.summary())
env = gym.make(env_name)

game = 100
for _ in range(game):
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    while not terminated and not truncated:
        action = agent.main_model.predict(obs.reshape(1, env.observation_space.shape[0]), verbose=0)
        dict1 = {
            "Left":action[0][0],
            "Right":action[0][1],
            "Up":action[0][2],
            "Down":action[0][3],
            "nothing":action[0][4]
        }
        # print(f"state:\nHedefe uzaklık X:{obs[0]}\t Hedefe uzaklık Y:{obs[1]}\nTopa uzaklık X:{obs[2]}\tTopa uzaklık Y:{obs[3]}\nTopun X yönü:{obs[4]}\t Topun Y yönü:{obs[5]}")
        
        """print(f"observation:{obs}action:{dict1}\t choosen action:{np.argmax(action)}")
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # Pencere kapatılırsa çık
                    waiting = False
                if event.type == pygame.KEYDOWN:  # Bir tuşa basıldığında
                    if event.key == pygame.K_SPACE:  # Eğer SPACE tuşuna basıldıysa
                        waiting = False
"""

        action = np.argmax(action)
        #pygame.time.delay(8000)
        obs, reward, terminated, truncated, info = env.step(action)
        if reward < 0:
            print(f"- reward var. obs:{obs}")
        total_reward += reward
        # print(f"Action uygulandı alınan reward:{reward}\ttoplam reward:{total_reward}, next obs:{obs}",end="\n"*3) 
        
        env.render()  




env.close()

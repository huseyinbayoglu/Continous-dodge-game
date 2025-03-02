import gymnasium as gym
from DQN import DQN
import numpy as np
import dodge_game_env  # DodgeGameEnv'nin doğru olduğundan emin ol
import pygame
import tensorflow as tf 
import os 
from tqdm import tqdm
# TODO Make a test function to test multiple environment and visualize the results

# TODO Make a script that can get the q values for special game state for example 
# Q values for being left of the target. or top of the ball going up


def general_test(model_path:str,n_env:int,step:int,env_name:str):
    result = {"Win_Games":0,"Loosing_Games":0,"Truncated_Games":0}
    path = os.path.join("models", model_path)
    if os.path.exists(path):
        model = tf.keras.models.load_model(path)
    else:
        print(f"model cannot be found")
    
    envs = [gym.make(env_name) for _ in range(n_env)]
    obss = [env.reset()[0] for env in envs]
    dones = np.array([False]*len(envs))

    for __1 in tqdm(range(1,step+1)):
        actions = model.predict(np.array(obss).reshape(len(obss),-1), verbose = False)
        actions = np.argmax(actions, axis=1)

        # Execute the action in every environment
        for idx,env in enumerate(envs):
            if not dones[idx]:
                next_obs, reward, terminated, _, _ = env.step(actions[idx]) 
                if terminated:
                    dones[idx] = True 
                    if reward > 0:
                        result["Win_Games"] += 1
                        env.close()
                    if reward < 0:
                        result["Loosing_Games"] += 1
                        env.close()
                if __1 == step and not terminated:
                    result["Truncated_Games"] += 1
                obss[idx] = next_obs
    return result 


print(general_test("deneme_son2.keras",500,200,"DodgeGame-v0"))
print(general_test("deneme_best2.keras",500,200,"DodgeGame-v0"))
    



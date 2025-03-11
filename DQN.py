import pandas as pd
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from dodge_game_env import DodgeGameEnv
from collections import deque
import gymnasium as gym 
import matplotlib.pyplot as plt 
import os 


# TODO results might be wrong. The agent plays perfectly but the result rewards not good.Fix it ✅ 

# TODO Make a wrapper to record game. And turn it to an mp4 file or save env state,and actions
# then replay it to get actual results

# TODO When obs == [0,0] then agent take negative reward action! When achive the target getting
# farer reward not achived reward!! so its not a good thing to achive the target!!

# TODO Visualize the number of winning game, loosing game, truncated game ✅

# TODO Make a function that analysis the replay buffer. ✅

# TODO Visualize the result of analyzing replay buffer ✅

# TODO Deploy the project using a cloud service when its done.


class DQN:
    def __init__(self, env_name, train_frequency: int = 4, n_env: int = 4, gamma=0.99, 
                 batch_size=32,epsilon_decay:float = .995, min_epsilon:float = .1,
                 update_target_frequency: int = 15, maxlen : int = 1_000_000) -> None:
        self.env_name = env_name  
        self.n_env = n_env  
        self.gamma = gamma  
        self.batch_size = batch_size
        self.train_frequency = train_frequency
        self.epsilon = .99
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self._update_target_frequency = update_target_frequency

        # SubprocVecEnv ile paralel environment'ları oluştur
        self.env = self.make_env()

        self.input_shape = self.env.observation_space.shape[0]  
        self.output_shape = self.env.action_space.n  

        self.main_model = self._get_model()  
        self.target_model = self._get_model()  
        self._update_target_model(tau=1)  # İlk başta hedef ağı güncelle

        self.maxlen = maxlen
        self.replay_buffer = deque(maxlen=self.maxlen)  # Deneyim havuzu

        self.max_ep_reward = np.inf

    def _get_model(self):
        """DQN Modeli"""
        """model = Sequential([
            Input(shape=(self.input_shape,)),  
            Dense(32, activation="relu"), 
            Dense(32, activation="relu"), 
            Dense(self.output_shape, activation="linear")  
        ])"""
        model = Sequential([
            Input(shape=(self.input_shape,)),  
            Dense(64, activation="relu"),  
            Dense(32, activation="relu"),  
            Dense(self.output_shape, activation="linear")  
        ])
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                      loss="mse", metrics=["accuracy"])
        return model

    def _update_target_model(self, tau=1):
      """Hedef ağı yumuşak güncelleme ile günceller."""
      target_weights = self.target_model.get_weights()
      main_weights = self.main_model.get_weights()
      for i in range(len(target_weights)):
          target_weights[i] = tau * main_weights[i] + (1 - tau) * target_weights[i]
      self.target_model.set_weights(target_weights)
     # self.target_model.set_weights(self.main_model.get_weights())

    def make_env(self):
        return gym.make(self.env_name)

    def rollout(self,n_step:int=50):
        envs = [self.make_env() for _ in range(self.n_env)]
        obss = np.array([env.reset()[0] for env in envs]).reshape((self.n_env, self.input_shape)) 
        dones = np.array([False] * self.n_env)

        terminated_game = {
            "winning_game":0,
            "loosing_game":0,
            "truncated_game":0
        }
        episode_lengths = []
        episode_rewards = []
        
        step_counts = np.zeros(self.n_env)  # Her env için adım sayacı
        total_rewards = np.zeros(self.n_env)  # Her env için ödül sayacı
        total_step_counter = 0
        for __1 in range(1,n_step+1): 
            actions = self.choose_action(obss=obss)
            for idx, env in enumerate(envs):
                # if the environment is not terminate
                if not dones[idx]: 
                    # Execute the action
                    next_obs, reward, terminated, _, _ = env.step(actions[idx]) 
                    self.replay_buffer.append((np.copy(obss[idx]), actions[idx], reward, np.copy(next_obs), terminated))
                    step_counts[idx] += 1  # Adım sayısını artır
                    total_rewards[idx] += reward  # Ödülü ekle
                    total_step_counter += 1
                    
                    if terminated:
                        episode_lengths.append(step_counts[idx])
                        episode_rewards.append(total_rewards[idx])
                        dones[idx] = True
                        if reward >= 0:
                            terminated_game["winning_game"] += 1
                            # for _ in range(700):
                                # self.replay_buffer.append((np.copy(obss[idx]), actions[idx], reward, np.copy(next_obs), terminated))
                        if reward <= 0:
                            terminated_game["loosing_game"] += 1 
                            # for _ in range(30):
                                #  self.replay_buffer.append((np.copy(obss[idx]), actions[idx], reward, np.copy(next_obs), terminated))

                    # if reward < 0:
                    #     print(f"Negative reward!!! obs:{obss[idx]}, reward:{reward}, action:{actions[idx]}")
                    #     print(f"modelin tahmini:{self.main_model.predict(np.array(obss[idx]).reshape(1,2))}")
                    if __1 == n_step and not terminated:
                        terminated_game["truncated_game"] += 1
                    obss[idx] = next_obs


        avg_episode_length = np.mean(np.array(episode_lengths)) if episode_lengths else 0
        avg_episode_reward = np.mean(np.array(episode_rewards)) if episode_rewards else 0
        max_episode_length = np.max(np.array(episode_lengths))  if episode_lengths else 0  
        min_episode_length = np.min(np.array(episode_lengths))  if episode_lengths else 0
        max_episode_reward = np.max(np.array(episode_rewards))  if episode_rewards else 0
        min_episode_reward = np.min(np.array(episode_rewards))  if episode_rewards else 0
        return avg_episode_length, avg_episode_reward, total_step_counter, max_episode_length, min_episode_length,max_episode_reward, min_episode_reward, terminated_game

    def train(self, step):
        """Deneyim havuzundan örnek alıp modeli eğitir"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Öğrenme oranını güncelle
        """if 250 <= step:
            tf.keras.backend.set_value(self.main_model.optimizer.lr, 0.003)  # Yeni öğrenme oranı
        elif 1000 <= step:
            tf.keras.backend.set_value(self.main_model.optimizer.lr, 0.001) 
        else:
            tf.keras.backend.set_value(self.main_model.optimizer.lr, 0.001)  # Varsayılan öğrenme oranı
            #tf.keras.backend.set_value(self.main_model.optimizer.lr, 0.003)  # Varsayılan öğrenme oranı
        """
        # print("TRAİN EDİLECEK")
        minibatch = [self.replay_buffer[np.random.randint(0, len(self.replay_buffer))] for _ in range(self.batch_size)]
        states, actions, rewards, next_states, dones = zip(*minibatch)

        counter_reward = 0
        counter_reward2 = 0
        for k in rewards:
            if k == 5:
                counter_reward += 1
            if k == -5:
                counter_reward2 += 1
        print(f"Bu stepte minibatchte kazanılan oyun durumu sayısı:{counter_reward}\
              Bu stepte minibatchte kaybedilen oyun durumu sayısı:{counter_reward2}")

        # print(f"Minibatch'ten ilk 1 experience\n{minibatch[:1]}")
        states = np.array(states)
        next_states = np.array(next_states)

        # Q-Değerleri hesapla
        q_values = self.main_model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        # print(f"Minibatch için Q değerleri\n{q_values[:1]}")
        # print(f"Minibatch için Next_Q değerleri\n{next_q_values[:1]}")
        # Hedef Q-Değerleri
        # terminals = []
        for i in range(self.batch_size):
            target_q = rewards[i]
            if not dones[i]:
                target_q += self.gamma * np.max(next_q_values[i])  # Bellman eşitliği

            """if dones[i]:
                print(f"terminal değer.Action:{actions[i]} önceki q değerleri:{q_values[i]}")
                terminals.append(i)"""

            q_values[i][actions[i]] = target_q  # Güncellenmiş hedef Q
            """if dones[i]:
                print(f"terminal değer. label q değerleri:{q_values[i]}",end="\n"*6)"""
        # Modeli eğit
        
        # print(f"Minibatch için label değerleri\n{q_values[:1]}")
        history = self.main_model.fit(states, q_values, epochs=1, verbose=1, batch_size=self.batch_size)
        # print(f"eğitim sonrası terminal q tahminleri: {self.main_model.predict(states)[terminals]}")
        # print(f"Eğitim sonrası q değer tahminleri \n{self.main_model.predict(np.array(states[:1]).reshape(1,2))}")
        return history.history['loss'][0], history.history["accuracy"][0]

    def learn(self, total_steps: int = 1000, plot:bool = False,n_step:int = 50):
        step = 0
        total_frame = 0
        episode_lengths = []  # Ortalama episode uzunluklarını sakla
        episode_rewards = []  # Ortalama episode ödüllerini sakla
        episode_max_lengths = []
        episode_min_lengths = []
        episode_max_rewards = []
        episode_min_rewards = []
        winning_game = []
        loosing_game = []
        truncated_game = []
        winning_frames = []
        closer_frames = []
        farer_frames = []
        loosing_frames = []
        losses = []  # Model kayıplarını sakla
        accuracies = []  
        recor_reward = - np.inf
        while step < total_steps:
            st1 = time.time()
            avg_ep_length, avg_ep_reward, adding_total_step, \
                max_ep_len, min_ep_len, max_ep_reward, min_ep_reward,\
                     terminated_game = self.rollout(n_step = n_step)
            episode_lengths.append(avg_ep_length)
            episode_rewards.append(avg_ep_reward)
            episode_max_lengths.append(max_ep_len)
            episode_min_lengths.append(min_ep_len)
            episode_max_rewards.append(max_ep_reward)
            episode_min_rewards.append(min_ep_reward)
            winning_game.append(terminated_game["winning_game"])
            loosing_game.append(terminated_game["loosing_game"])
            truncated_game.append(terminated_game["truncated_game"])
            replay_buffer_analysis = self.analyze_replay_buffer()
            winning_frames.append(replay_buffer_analysis["winning_state"])
            closer_frames.append(replay_buffer_analysis["closer_state"])
            farer_frames.append(replay_buffer_analysis["farer_state"])
            loosing_frames.append(replay_buffer_analysis["collision_state"])
            if terminated_game["loosing_game"] != 0:
                if terminated_game["winning_game"]/terminated_game["loosing_game"] > recor_reward:
                    self.save_model("deneme_best2.keras")
            else:
                if terminated_game["winning_game"]/(terminated_game["loosing_game"]+.1) > recor_reward:
                    self.save_model("deneme_best2.keras")
            step += 1
            total_frame += adding_total_step
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            print(f"""Güncel step:{step}\nAverage episode length:{avg_ep_length}\nAverage reward:{avg_ep_reward}\nmax episode length:{max_ep_len}\tmin episode length:{min_ep_len}\nmax episode reward:{max_ep_reward}\tmin episode reward{min_ep_reward}\ntotal_frame:{total_frame}\nfps:{round(adding_total_step/(time.time()-st1))}\nepsilon:{self.epsilon}\n{terminated_game}\n Replay buffer results:{replay_buffer_analysis}""",end="\n"*4)
            self.save_model("deneme_son2.keras")
            if step % self.train_frequency == 0:
                loss, accuracy = self.train(step)
                if loss is not None:
                    losses.append(loss)
                if accuracy is not None:
                    accuracies.append(accuracy)

            if step % self._update_target_frequency == 0:
                self._update_target_model()
            
            
            # Belirli aralıklarla grafikleri çiz
            if plot:  
                self.plot_metrics(episode_lengths, episode_rewards, losses, accuracies,episode_max_lengths,
                                  episode_min_lengths,episode_max_rewards,episode_min_rewards,False,
                                  winning_game = winning_game,loosing_game = loosing_game,
                                  truncated_game = truncated_game,winning_frame=winning_frames,
                                  closer_frame=closer_frames, farer_frame=farer_frames, loosing_frame = loosing_frames)
            if step >= total_steps:
                self.plot_metrics(episode_lengths, episode_rewards, losses, accuracies,episode_max_lengths,
                                  episode_min_lengths,episode_max_rewards,episode_min_rewards,True,
                                  winning_game = winning_game,loosing_game = loosing_game,
                                  truncated_game = truncated_game,winning_frame=winning_frames,
                                  closer_frame=closer_frames, farer_frame=farer_frames, loosing_frame = loosing_frames)
            
    def choose_action(self, obss):
        if np.random.random() > self.epsilon:
            actions = self.main_model.predict(obss, verbose=0)  # Tek seferde inference
            actions = np.argmax(actions, axis=1)  # En iyi aksiyonu seç
        else:
            actions = np.random.randint(0, self.output_shape, size=len(obss))  # Rastgele aksiyon seç
        
        return np.array(actions).reshape((self.n_env, 1))
    
    def save_model(self, filename: str):
        folder = "models"
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, filename)
        self.main_model.save(path)
        # print(f"Model saved at {path}")
    
    def load_model(self, filename: str):
        path = os.path.join("models", filename)
        if os.path.exists(path):
            self.main_model = tf.keras.models.load_model(path)
            self._update_target_model()
            # print(f"Model loaded from {path}")
        else:
            print("Error: Model file does not exist.")

    def plot_metrics(self, episode_lengths, episode_rewards, losses,accuracies, max_len, 
                     min_len, max_rew, min_rew,show=False,winning_game = [],loosing_game = [],
                     truncated_game = [], winning_frame = [], closer_frame = [], 
                     farer_frame = [], loosing_frame = []):
        def moving_average(data, window_size=5):
            if len(data) < window_size:
                return np.convolve(data, np.ones(len(data)) / len(data), mode='valid')
            return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

        if not hasattr(self, 'fig'):
            # İlk kez çalışıyorsa figür ve çizgileri oluştur
            self.fig, self.axs = plt.subplots(2, 3, figsize=(18, 6))

            # Kayıp grafiği (Loss)
            self.loss_line, = self.axs[0, 0].plot([], [], label="Loss", color='r')
            self.axs[0, 0].set_xlabel("Training Steps")
            self.axs[0, 0].set_ylabel("Loss")
            self.axs[0, 0].set_title("Model Loss")
            self.axs[0, 0].legend()

            # Accuracy grafiği
            self.accuracy_line, = self.axs[0, 1].plot([], [], label="Accuracy", color='b')
            self.axs[0, 1].set_xlabel("Training Steps")
            self.axs[0, 1].set_ylabel("Accuracy")
            self.axs[0, 1].set_title("Model Accuracy")
            self.axs[0, 1].legend()

            # Episode Length Grafiği
            self.length_line, = self.axs[1, 0].plot([], [], label="Avg Episode Length", color='g')
            self.length_avg_line, = self.axs[1, 0].plot([], [], label="Length (Moving Avg)", linewidth=3, color='darkgreen')

            self.axs[1,0].set_xlabel("Training Steps")
            self.axs[1,0].set_ylabel("Frames per Episode")
            self.axs[1,0].set_title("Average Episode Length")
            self.axs[1,0].legend()

            # Reward Grafiği
            self.reward_line, = self.axs[1, 1].plot([], [], label="Avg Episode Reward", color='b')
            self.reward_avg_line, = self.axs[1, 1].plot([], [], label="Reward (Moving Avg)", linewidth=3, color='darkblue')

            self.axs[1, 1].set_xlabel("Training Steps")
            self.axs[1, 1].set_ylabel("Reward")
            self.axs[1, 1].set_title("Average Episode Reward")
            self.axs[1, 1].legend()

            self.win_line, = self.axs[0, 2].plot([], [], label="Winning Games", color='green')
            self.lose_line, = self.axs[0, 2].plot([], [], label="Losing Games", color='red')
            self.trunc_line, = self.axs[0, 2].plot([], [], label="Truncated Games", color='orange')
            self.axs[0, 2].set_xlabel("Training Steps")
            self.axs[0, 2].set_ylabel("Game Count")
            self.axs[0, 2].set_title("Terminated Games")
            self.axs[0, 2].legend()

            # self.wining_frame_line, = self.axs[1, 2].plot([], [], label="Winning state frame", color='green')
            self.closer_frame_line, = self.axs[1, 2].plot([], [], label="Closer state number", color='blue')
            self.farer_frame_line, = self.axs[1, 2].plot([], [], label="farer state number", color='orange')
            # self.loosing_frame_line, = self.axs[1, 2].plot([], [], label="loosing state number", color='red')
            self.axs[1, 2].set_xlabel("Training Steps")
            self.axs[1, 2].set_ylabel("Frame Count")
            self.axs[1, 2].set_title("Replay Buffer")
            self.axs[1, 2].legend()


            plt.ion()  # Interactive mode
            plt.show()

        # X eksenini oluştur
        x_data = np.arange(len(episode_lengths))

        # Çizgileri güncelle
        self.loss_line.set_xdata(x_data[:len(losses)])
        self.loss_line.set_ydata(losses)

        self.accuracy_line.set_xdata(x_data[:len(accuracies)])
        self.accuracy_line.set_ydata(accuracies)

        self.length_line.set_xdata(x_data)
        self.length_line.set_ydata(episode_lengths)

        self.reward_line.set_xdata(x_data)
        self.reward_line.set_ydata(episode_rewards)


        window_size = 5  # Hareketli ortalama penceresi
        # Hareketli ortalama hesapla
        ma_lengths = moving_average(episode_lengths,window_size=window_size)
        ma_rewards = moving_average(episode_rewards,window_size=window_size)

        # Hareketli ortalamaları güncelle
        

        # X eksenini pencere boyutundan başlayarak güncelle
        """self.length_avg_line.set_xdata(x_data[window_size - 1 : len(ma_lengths) + window_size - 1])
        self.length_avg_line.set_ydata(ma_lengths)

        self.reward_avg_line.set_xdata(x_data[window_size - 1 : len(ma_rewards) + window_size - 1])
        self.reward_avg_line.set_ydata(ma_rewards)"""

        # Winning, losing ve truncated oyunları güncelle
        self.win_line.set_xdata(x_data[:len(winning_game)])
        self.win_line.set_ydata(winning_game)
        self.lose_line.set_xdata(x_data[:len(loosing_game)])
        self.lose_line.set_ydata(loosing_game)
        self.trunc_line.set_xdata(x_data[:len(truncated_game)])
        self.trunc_line.set_ydata(truncated_game)

        # Frame sayılarını günceller
        """self.wining_frame_line.set_xdata(x_data[:len(winning_frame)])
        self.wining_frame_line.set_ydata(winning_frame)"""

        self.closer_frame_line.set_xdata(x_data[:len(closer_frame)])
        self.closer_frame_line.set_ydata(closer_frame)

        self.farer_frame_line.set_xdata(x_data[:len(farer_frame)])
        self.farer_frame_line.set_ydata(farer_frame)

        """self.loosing_frame_line.set_xdata(x_data[:len(loosing_frame)])
        self.loosing_frame_line.set_ydata(loosing_frame)"""

        # fill_between için öncekini temizle ve yeniden çiz
        for coll in self.axs[1, 0].collections:
            coll.remove()
        for coll in self.axs[1, 1].collections:
            coll.remove()

        # Min-max değerleri kullanarak dolgu alanlarını belirle
        self.axs[1, 0].fill_between(x_data, min_len, max_len, color='green', alpha=0.2, label="Length Range")
        self.axs[1, 1].fill_between(x_data, min_rew, max_rew, color='blue', alpha=0.2, label="Reward Range")

        # === Y ekseni sınırlarını akıllıca ayarlama ===
        def set_ylim_smart(ax, data_min, data_max):
            padding = (data_max - data_min) * 0.1  # %10 boşluk ekleyelim
            if padding == 0:  # Eğer tüm değerler aynıysa küçük bir boşluk ekleyelim
                padding = max(0.5, abs(data_max) * 0.1)
            ax.set_ylim(data_min - padding, data_max + padding)

        # Y eksenini güncelle
        set_ylim_smart(self.axs[1, 0], min(min_len), max(max_len))
        set_ylim_smart(self.axs[1, 1], min(min_rew), max(max_rew))

        # Eksenleri güncelle
        for ax in self.axs.flatten():
            ax.relim()
            ax.autoscale_view()


        plt.draw()
        plt.pause(0.1)
        if show:
            plt.savefig("Results.png")

    def analyze_replay_buffer(self):
        rewards = np.fromiter((r for _,_,r,_,_ in self.replay_buffer),dtype=np.float32)
        count_of_winning_states = np.count_nonzero(rewards == 5)
        count_of_closer_states = np.count_nonzero(rewards == -.1)
        count_of_farer_states = np.count_nonzero(rewards == -1.)
        count_of_collision_states = np.count_nonzero(rewards == -5)
        return {
            "winning_state":count_of_winning_states,
            "closer_state":count_of_closer_states,
            "farer_state":count_of_farer_states,
            "collision_state":count_of_collision_states,
            "Size": len(self.replay_buffer)
        }

    
if __name__ == "__main__":
    import time 
    env_name = "DodgeGame-v0"  # Kendi ortamının adını buraya gir
    agent = DQN(env_name,n_env=100,train_frequency=1,maxlen=500_000,update_target_frequency=5,
                batch_size=128,gamma=1, epsilon_decay=.975,min_epsilon=.1)
    agent.load_model("deneme_son2.keras")
    agent.target_model.set_weights(agent.main_model.get_weights())
    agent.epsilon = 0.1
    st = time.time()
    agent.learn(total_steps=100,plot=True,n_step=100)
    print(f"Training time: {time.time()-st}")
    agent.save_model("deneme_son2.keras")




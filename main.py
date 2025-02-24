import gymnasium as gym
from stable_baselines3 import PPO
import torch
import dodge_game_env
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3.common.monitor import Monitor

# GPU kontrolü
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} for training.")

# Log klasörünü oluştur
log_dir = "env_logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "monitor.csv")

# Ortamı oluştur ve monitor ile sar
env = gym.make("DodgeGame-v0")
env = Monitor(env, filename=log_file)

# Sinir ağı yapısını belirle
policy_kwargs = dict(net_arch=[128, 64])

# PPO Modeli oluştur
model_ppo = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    device=device, 
    learning_rate=7e-5,  
    batch_size=128,  
    policy_kwargs=policy_kwargs
)

# Modeli eğit
learning_time_steps = 200_000
st = time.time()
model_ppo.learn(total_timesteps=learning_time_steps, log_interval=4, progress_bar=True)
model_ppo.save("ppo_yeni")

print(f"Eğitim tamamlandı. Geçen süre: {time.time() - st}")

# Eğitim sonrası verileri görselleştir
def plot_results(log_file):
    if not os.path.exists(log_file):
        print("Log dosyası bulunamadı.")
        return

    monitor_data = pd.read_csv(log_file, skiprows=1)
    monitor_data = monitor_data.dropna(subset=["r", "l"])  # Eksik verileri temizle

    if len(monitor_data) == 0:
        print("Görselleştirilecek veri yok.")
        return

    episodes = range(len(monitor_data))
    episode_rewards = monitor_data["r"]
    episode_lengths = monitor_data["l"]

    # 🔹 Hareketli ortalama hesapla (Son 15 episode)
    avg_rewards = episode_rewards.rolling(window=15).mean()
    avg_lengths = episode_lengths.rolling(window=15).mean()

    # Grafik çiz
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Grafik: Episode vs. Toplam Ödül
    axes[0].plot(episodes, episode_rewards, label="Toplam Ödül", color="b", alpha=0.6)
    axes[0].plot(episodes, avg_rewards, label="15 Episode Hareketli Ortalama", color="g")
    axes[0].set_title("Episode vs. Toplam Ödül")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Toplam Ödül")
    axes[0].legend()
    axes[0].grid(True)

    # 2. Grafik: Episode vs. Episode Uzunluğu
    axes[1].plot(episodes, episode_lengths, label="Episode Uzunluğu", color="r", alpha=0.6)
    axes[1].plot(episodes, avg_lengths, label="15 Episode Hareketli Ortalama", color="m")
    axes[1].set_title("Episode vs. Episode Uzunluğu")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Uzunluğu")
    axes[1].legend()
    axes[1].grid(True)

    plt.show()

# Grafikleri eğitim sonrası göster
plot_results(log_file)

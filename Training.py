# 이 코드는 /home/rl/RL/Code/RL Study 에 존재합니다.

from gymnasium_robotics import register_robotics_envs
register_robotics_envs()

import pathlib
import gymnasium as gym
from stable_baselines3 import PPO as Algo # 알고리즘 변경 시 수정
from stable_baselines3.common.logger import configure
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from typing import Any, Dict

import torch as th

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
import imageio

CWD = pathlib.Path(__file__).absolute().parent.parent.parent # /home/rl/RL

# 알고리즘
algorithm = "PPO"
# 환경 이름
environment = "Reach"
# 모델 이름
model_name = "FetchReachDense-v2"
# 불러올 모델 넘버
load_model_number = "0000"

models_dir = f"{CWD}/Result/RL Study/{environment}/model/"
log_dir = f"{CWD}/Result/RL Study/{environment}/logs/{algorithm}" # {environment}폴더 위치로 가서 터미널을 연 뒤, "tensorboard --logdir=logs" 명령어 실행
log_img_dir = f"{CWD}/Result/RL Study/{environment}/logs/img/"
model_path = f"{models_dir}/{environment}_{algorithm}_{load_model_number}"
algorithm_time_path = f"{CWD}/Result/RL Study/{environment}/logs/{algorithm}/time.txt"

if not os.path.exists(models_dir):
   os.makedirs(models_dir)

if not os.path.exists(log_dir):
   os.makedirs(log_dir)

if not os.path.exists(log_img_dir):
   os.makedirs(log_img_dir)

# set up logger
train_logger = configure(log_dir, ["stdout", "csv", "log", "tensorboard"])

#--------------------------------------------------#
#  Training...                                     #
#--------------------------------------------------#
#"""
env = gym.make(model_name, render_mode="rgb_array") # 환경 변경 시 수정
env.reset()

#model = Algo("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)
model = Algo("MultiInputPolicy", env, verbose=1)
# Set logger
model.set_logger(train_logger)

TIMESTEPS = 10000

total_time = 0

for i in range(200):
   start = time.time()
   model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
   model.save(f"{models_dir}/{environment}_{algorithm}_{TIMESTEPS*(i+1)}")
   end = time.time()
   sub_time = end - start
   with open(algorithm_time_path, 'a') as file:
      file.write(f"{TIMESTEPS*(i+1)} 훈련 : {sub_time:.5f} sec\n")
   total_time += sub_time

with open(algorithm_time_path, 'a') as file:
   file.write(f"전체 훈련 시간 : {total_time:.5f} sec\n\n")
#"""

#--------------------------------------------------#
#  Additional Training...                          #
#--------------------------------------------------#
"""
env = gym.make(model_name, render_mode="human")
env.reset()

model = Algo.load(model_path, env=env)

# Set logger
model.set_logger(train_logger)

TIMESTEPS = 10000

total_time = 0

for i in range(20):
   start = time.time()
   model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"{algorithm}")
   model.save(f"{models_dir}/{environment}_{algorithm}_{TIMESTEPS*(i+1)+int(load_model_number)}")
   end = time.time()
   sub_time = end - start
   with open(algorithm_time_path, 'a') as file:
      file.write(f"{TIMESTEPS*(i+1)+int(load_model_number)} 훈련 : {sub_time:.5f} sec\n")
   total_time += sub_time

with open(algorithm_time_path, 'a') as file:
   file.write(f"전체 훈련 시간 : {total_time:.5f} sec\n\n")
#"""

#--------------------------------------------------#
#  Testing...                                      #
#--------------------------------------------------#
"""
env = gym.make(model_name, render_mode="human")
model = Algo.load(model_path, env=env)

observation, info = env.reset()

for _ in range(1000): 
   action, _ = model.predict(observation)
   observation, reward, terminated, truncated, info = env.step(action)
   
   #image = env.render()
   #test_logger = configure(test_log_dir, Image(image, "HWC"))

   if terminated or truncated:
      observation, info = env.reset()

env.close()
#"""
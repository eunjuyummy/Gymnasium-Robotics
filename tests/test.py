'''
import gymnasium as gym
from stable_baselines3 import PPO# env = gym.make("CartPole-v1", render_mode='human')
# env = gym.make("FetchSlide-v2", render_mode='human')
# env = gym.make("FetchPickAndPlace-v2", render_mode='human')
# env = gym.make("FetchReach-v2", render_mode='human')
#env = gym.make("FetchPush-v2", render_mode='rgb_array')
env = gym.make('FetchSlideDense-v2')
#env = gym.make("HumanoidStandup-v4", render_mode='rgb_array')
model = PPO("MultiInputPolicy", env, verbose=1)
#model = PPO("MlpPolicy", env, verbose=1)model.learn(total_timesteps=1000)
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render(mode='human')
env.close()
'''
import gymnasium as gym

env = gym.make("FetchReach-v2")
env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

# The following always has to hold:
assert reward == env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
assert truncated == env.compute_truncated(obs["achieved_goal"], obs["desired_goal"], info)
assert terminated == env.compute_terminated(obs["achieved_goal"], obs["desired_goal"], info)

# However goals can also be substituted:
substitute_goal = obs["achieved_goal"].copy()
substitute_reward = env.compute_reward(obs["achieved_goal"], substitute_goal, info)
substitute_terminated = env.compute_terminated(obs["achieved_goal"], substitute_goal, info)
substitute_truncated = env.compute_truncated(obs["achieved_goal"], substitute_goal, info)
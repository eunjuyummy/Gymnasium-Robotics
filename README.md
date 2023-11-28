[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <img src="https://raw.githubusercontent.com/Farama-Foundation/Gymnasium-Robotics/main/gymrobotics-revised-text.png" width="500px"/>
</p>

This library contains a collection of Reinforcement Learning robotic environments that use the [Gymansium](https://gymnasium.farama.org/) API. The environments run with the [MuJoCo](https://mujoco.org/) physics engine and the maintained [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html).

The documentation website is at [robotics.farama.org](https://robotics.farama.org/), and we have a public discord server (which we also use to coordinate development work) that you can join here: [https://discord.gg/YymmHrvS](https://discord.gg/YymmHrvS)

## Environments

`Gymnasium-Robotics` includes the following groups of environments:

* [Fetch](https://robotics.farama.org/envs/fetch/) - A collection of environments with a 7-DoF robot arm that has to perform manipulation tasks such as Reach, Push, Slide or Pick and Place.

## Multi-goal API

The robotic environments use an extension of the core Gymansium API by inheriting from [GoalEnv](https://robotics.farama.org/envs/#) class. The new API forces the environments to have a dictionary observation space that contains 3 keys:

* `observation` - The actual observation of the environment
* `desired_goal` - The goal that the agent has to achieved
* `achieved_goal` - The goal that the agent has currently achieved instead. The objective of the environments is for this value to be close to `desired_goal`

This API also exposes the function of the reward, as well as the terminated and truncated signals to re-compute their values with different goals. This functionality is useful for algorithms that use Hindsight Experience Replay (HER).

The following example demonstrates how the exposed reward, terminated, and truncated functions
can be used to re-compute the values with substituted goals. The info dictionary can be used to store
additional information that may be necessary to re-compute the reward, but that is independent of the
goal, e.g. state derived from the simulation.

```python
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
```

The `GoalEnv` class can also be used for custom environments.

## Citation

If you use this in your research, please cite:
```
@software{gymnasium_robotics2023github,
  author = {Rodrigo de Lazcano and Kallinteris Andreas and Jun Jet Tai and Seungjae Ryan Lee and Jordan Terry},
  title = {Gymnasium Robotics},
  url = {http://github.com/Farama-Foundation/Gymnasium-Robotics},
  version = {1.2.3},
  year = {2023},
}
```

## Action Space

The action space is a `Box(-1.0, 1.0, (4,), float32)`. An action represents the Cartesian displacement dx, dy, and dz of the end effector. In addition to a last action that controls closing and opening of the gripper.

| Num | Action                                                                                                                                |
| --- |---------------------------------------------------------------------------------------------------------------------------------------|
| 0   | Displacement of the end effector in the x direction dx                                                                                |
| 1   | Displacement of the end effector in the y direction dy                                                                                |
| 2   | Displacement of the end effector in the z direction dz                                                                                |
| 3   | Positional displacement per timestep of each finger of the gripper                                                                    |

## Observation Space

The observation is a `goal-aware observation space`. It consists of a dictionary with information about the robot's end effector state and goal. The kinematics observations are derived from Mujoco bodies known as [sites](https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=site#body-site) attached to the body of interest such as the block or the end effector.
Only the observations from the gripper fingers are derived from joints. Also to take into account the temporal influence of the step time, velocity values are multiplied by the step time dt=number_of_sub_steps*sub_step_time. The dictionary consists of the following 3 keys:

* `observation`: its value is an `ndarray` of shape `(25,)`. It consists of kinematic information of the block object and gripper. The elements of the array correspond to the following:
  
| Num | Observation                                                                                                                           |                     
|-----|---------------------------------------------------------------------------------------------------------------------------------------|
| 0   | End effector x position in global coordinates                                                                                         |
| 1   | End effector y position in global coordinates                                                                                         | 
| 2   | End effector z position in global coordinates                                                                                         |                              
| 3   | Block x position in global coordinates                                                                                                |                                   
| 4   | Block y position in global coordinates                                                                                                |                           
| 5   | Block z position in global coordinates                                                                                                |                                 
| 6   | Relative block x position with respect to gripper x position in globla coordinates. Equals to x<sub>gripper</sub> - x<sub>block</sub> |                                 
| 7   | Relative block y position with respect to gripper y position in globla coordinates. Equals to y<sub>gripper</sub> - y<sub>block</sub> |                                
| 8   | Relative block z position with respect to gripper z position in globla coordinates. Equals to z<sub>gripper</sub> - z<sub>block</sub> |                          
| 9   | Joint displacement of the right gripper finger                                                                                        |      
| 10  | Joint displacement of the left gripper finger                                                                                         |     
| 11  | Global x rotation of the block in a XYZ Euler frame rotation                                                                          |                          
| 12  | Global y rotation of the block in a XYZ Euler frame rotation                                                                          |                
| 13  | Global z rotation of the block in a XYZ Euler frame rotation                                                                          |                        
| 14  | Relative block linear velocity in x direction with respect to the gripper                                                             |                     
| 15  | Relative block linear velocity in y direction with respect to the gripper                                                             |                             
| 16  | Relative block linear velocity in z direction                                                                                         |                             
| 17  | Block angular velocity along the x axis                                                                                               |                            
| 18  | Block angular velocity along the y axis                                                                                               |                         
| 19  | Block angular velocity along the z axis                                                                                               |                           
| 20  | End effector linear velocity x direction                                                                                              |                                
| 21  | End effector linear velocity y direction                                                                                              |                        
| 22  | End effector linear velocity z direction                                                                                              |                         
| 23  | Right gripper finger linear velocity                                                                                                  |  
| 24  | Left gripper finger linear velocity                                                                                                   |   

* `desired_goal`: this key represents the final goal to be achieved. In this environment it is a 3-dimensional `ndarray`, `(3,)`, that consists of the three cartesian coordinates of the desired final block position `[x,y,z]`. In order for the robot to perform a pick and place trajectory, the goal position can be elevated over the table or on top of the table. The elements of the array are the following:

| Num | Observation                                                                                                                           | 
|-----|---------------------------------------------------------------------------------------------------------------------------------------|
| 0   | Final goal block position in the x coordinate                                                                                         |
| 1   | Final goal block position in the y coordinate                                                                                         | 
| 2   | Final goal block position in the z coordinate                                                                                         |
  
* `achieved_goal`: this key represents the current state of the block, as if it would have achieved a goal. This is useful for goal orientated learning algorithms such as those that use [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (HER). The value is an `ndarray` with shape `(3,)`. The elements of the array are the following:

| Num | Observation                                                                                                                           | 
|-----|---------------------------------------------------------------------------------------------------------------------------------------|
| 0   | Current block position in the x coordinate                                                                                            | 
| 1   | Current block position in the y coordinate                                                                                            |
| 2   | Current block position in the z coordinate                                                                                            | 


## Rewards

The reward can be initialized as `sparse` or `dense`:
- *sparse*: the returned reward can have two values: `-1` if the block hasn't reached its final target position, and `0` if the block is in the final target position (the block is considered to have reached the goal if the Euclidean distance between both is lower than 0.05 m).
- *dense*: the returned reward is the negative Euclidean distance between the achieved goal position and the desired goal.

To initialize this environment with one of the mentioned reward functions the type of reward must be specified in the id string when the environment is initialized. For `sparse` reward the id is the default of the environment, `FetchPickAndPlace-v2`. However, for `dense` reward the id must be modified to `FetchPickAndPlaceDense-v2` and initialized as follows:

```python
import gymnasium as gym

env = gym.make('FetchPickAndPlaceDense-v2')
```

## Starting State

When the environment is reset the gripper is placed in the following global cartesian coordinates `(x,y,z) = [1.3419 0.7491 0.555] m`, and its orientation in quaternions is `(w,x,y,z) = [1.0, 0.0, 1.0, 0.0]`. The joint positions are computed by inverse kinematics internally by MuJoCo. The base of the robot will always be fixed at `(x,y,z) = [0.405, 0.48, 0]` in global coordinates.

The block's position has a fixed height of `(z) = [0.42] m ` (on top of the table). The initial `(x,y)` position of the block is the gripper's x and y coordinates plus an offset sampled from a uniform distribution with a range of `[-0.15, 0.15] m`. Offset samples are generated until the 2-dimensional Euclidean distance from the gripper to the block is greater than `0.1 m`.
The initial orientation of the block is the same as for the gripper, `(w,x,y,z) = [1.0, 0.0, 1.0, 0.0]`.

Finally the target position where the robot has to move the block is generated. The target can be in mid-air or over the table. The random target is also generated by adding an offset to the initial grippers position `(x,y)` sampled from a uniform distribution with a range of `[-0.15, 0.15] m`.
The height of the target is initialized at `(z) = [0.42] m ` and an offset is added to it sampled from another uniform distribution with a range of `[0, 0.45] m`.


## Episode End

The episode will be `truncated` when the duration reaches a total of `max_episode_steps` which by default is set to 50 timesteps.
The episode is never `terminated` since the task is continuing with infinite horizon.

## Arguments

To increase/decrease the maximum number of timesteps before the episode is `truncated` the `max_episode_steps` argument can be set at initialization. The default value is 50. For example, to increase the total number of timesteps to 100 make the environment as follows:

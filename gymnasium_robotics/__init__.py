# noqa: D104
from gymnasium.envs.registration import register

from gymnasium_robotics.core import GoalEnv
from gymnasium_robotics.envs.point_maze import maps


def register_robotics_envs():
    """Register all environment ID's to Gymnasium."""

    def _merge(a, b):
        a.update(b)
        return a

    for reward_type in ["sparse", "dense"]:
        suffix = "Dense" if reward_type == "dense" else ""
        kwargs = {
            "reward_type": reward_type,
        }

        # Fetch
        register(
            id=f"FetchSlide{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoPyFetchSlideEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchSlide{suffix}-v2",
            entry_point="gymnasium_robotics.envs:MujocoFetchSlideEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchPickAndPlace{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoPyFetchPickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchPickAndPlace{suffix}-v2",
            entry_point="gymnasium_robotics.envs:MujocoFetchPickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchReach{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoPyFetchReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchReach{suffix}-v2",
            entry_point="gymnasium_robotics.envs:MujocoPyFetchReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchReach{suffix}-v3",
            entry_point="gymnasium_robotics.envs:MujocoFetchReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchPush{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoPyFetchPushEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchPush{suffix}-v2",
            entry_point="gymnasium_robotics.envs:MujocoFetchPushEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        # Hand
        register(
            id=f"HandReach{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"HandReach{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"HandManipulateBlockRotateZ{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateZ{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateZ_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateZ_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateZ_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateZ_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockFull{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockFull{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        # Alias for "Full"
        register(
            id=f"HandManipulateBlock{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlock{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlock_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlock_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlock_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlock_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggFull{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggFull{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        # Alias for "Full"
        register(
            id=f"HandManipulateEgg{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEgg{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEgg_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEgg_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEgg_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEgg_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenFull{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenFull{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs:MujocoPyHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        #####################
        # D4RL Environments #
        #####################

        # ----- AntMaze -----

        register(
            id=f"AntMaze_UMaze{suffix}-v3",
            entry_point="gymnasium_robotics.envs:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.U_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Open{suffix}-v3",
            entry_point="gymnasium_robotics.envs:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Open_Diverse_G{suffix}-v3",
            entry_point="gymnasium_robotics.envs:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )
        register(
            id=f"AntMaze_Open_Diverse_GR{suffix}-v3",
            entry_point="gymnasium_robotics.envs:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Medium{suffix}-v3",
            entry_point="gymnasium_robotics.envs:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Medium_Diverse_G{suffix}-v3",
            entry_point="gymnasium_robotics.envs:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Medium_Diverse_GR{suffix}-v3",
            entry_point="gymnasium_robotics.envs:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Large{suffix}-v3",
            entry_point="gymnasium_robotics.envs:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Large_Diverse_G{suffix}-v3",
            entry_point="gymnasium_robotics.envs:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Large_Diverse_GR{suffix}-v3",
            entry_point="gymnasium_robotics.envs:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        # ----- PointMaze -----

        register(
            id=f"PointMaze_UMaze{suffix}-v3",
            entry_point="gymnasium_robotics.envs:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.U_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=300,
        )

        register(
            id=f"PointMaze_Open{suffix}-v3",
            entry_point="gymnasium_robotics.envs:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN,
                },
                kwargs,
            ),
            max_episode_steps=300,
        )

        register(
            id=f"PointMaze_Open_Diverse_G{suffix}-v3",
            entry_point="gymnasium_robotics.envs:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=300,
        )

        register(
            id=f"PointMaze_Open{suffix}-v3",
            entry_point="gymnasium_robotics.envs:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN,
                },
                kwargs,
            ),
            max_episode_steps=300,
        )

        register(
            id=f"PointMaze_Medium{suffix}-v3",
            entry_point="gymnasium_robotics.envs:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=600,
        )

        register(
            id=f"PointMaze_Medium_Diverse_G{suffix}-v3",
            entry_point="gymnasium_robotics.envs:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=600,
        )

        register(
            id=f"PointMaze_Medium_Diverse_GR{suffix}-v3",
            entry_point="gymnasium_robotics.envs:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=600,
        )

        register(
            id=f"PointMaze_Large{suffix}-v3",
            entry_point="gymnasium_robotics.envs:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=800,
        )

        register(
            id=f"PointMaze_Large_Diverse_G{suffix}-v3",
            entry_point="gymnasium_robotics.envs:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=800,
        )

        register(
            id=f"PointMaze_Large_Diverse_GR{suffix}-v3",
            entry_point="gymnasium_robotics.envs:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=800,
        )
        
        
    register(
        id="AdroitHandDoor-v0",
        entry_point="gymnasium_robotics.envs:AdroitHandDoorEnv",
        max_episode_steps=200,
    )

    register(
        id="AdroitHandHammer-v0",
        entry_point="gymnasium_robotics.envs:AdroitHandHammerEnv",
        max_episode_steps=200,
    )

    register(
        id="AdroitHandPen-v0",
        entry_point="gymnasium_robotics.envs:AdroitHandPenEnv",
        max_episode_steps=200,
    )

    register(
        id="AdroitHandRelocate-v0",
        entry_point="gymnasium_robotics.envs:AdroitHandRelocateEnv",
        max_episode_steps=200,
    )


__version__ = "1.1.0"

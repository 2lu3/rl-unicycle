import pybullet as p
import pybullet_data
from typing import List, Tuple, Optional
import gym
from gym.spaces import Box
import numpy as np
import time


class UniCycleEnv(gym.Env):
    p_goal = np.array([10, 10, 0])
    p_start = np.array([0, 0, 0])
    p_robot = p_start.copy()

    max_velocity = 10
    time_step: float = 0.01

    metadata = {"render.modes": ["ansi"]}
    action_space = Box(low=max_velocity, high=max_velocity, shape=(4,))  # 左足、右足、前後、左右
    reward_range: Tuple[float, float] = (-1, 100)
    observation_space: Box = Box(low=-20, high=20, shape=(4,))  # x, y, 前後, 左右

    def __init__(self, visualize=True):
        self.physicsClient: int
        if visualize == True:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        p.setGravity(0, 0, -10)
        p.setTimeStep(self.time_step)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id: int = p.loadURDF("plane.urdf")
        self.unicycle_id: int = p.loadURDF("./model/unicycle.urdf")

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        When end of episode is reached, you are responsible for calling `reset()` to reset this environment's state

        Args:
            action (object): an action provided by the agent

        Returns:
            Observation (object): agent's observation of the current environment
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() call will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debuggin, and sometimes learning)
        """

        p.stepSimulation()


def main():
    env = UniCycleEnv()
    for i in range(1000):
        env.step(None)
        time.sleep(0.1)


if __name__ == "__main__":
    main()

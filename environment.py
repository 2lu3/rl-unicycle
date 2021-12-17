import pybullet as p
import pybullet_data
from typing import List, Tuple
import gym
from gym.spaces import Box
import numpy as np
import time


class UnicycleEnv(gym.Env):
    p_goal: np.ndarray = np.array([10, 10, 0])
    p_start: np.ndarray = np.array([0, 0, 0])
    p_robot: np.ndarray = p_start.copy()
    p_saddle: np.ndarray = p_start.copy()

    joints_id: dict = dict()

    max_force = 30
    time_step: float = 0.05

    metadata = {"render.modes": ["ansi"]}
    action_space = Box(low=-max_force, high=max_force, shape=(4,))
    reward_range: Tuple[float, float] = (-1, 100)
    observation_space: Box = Box(low=-20, high=20, shape=(5,))

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

        for i in range(p.getNumJoints(self.unicycle_id)):
            info = p.getJointInfo(self.unicycle_id, i)
            joint_name = info[1]
            self.joints_id[joint_name.decode("utf-8")] = i
        print(self.joints_id)

    def step(self, action: List):
        """Run one timestep of the environment's dynamics.
        When end of episode is reached, you are responsible for calling `reset()` to reset this environment's state

        Args:
            action (object): 左足の力、右足の力、サドルの前後方向の力、サドルの左右方向の力

        Returns:
            Observation (object): 一輪車のx座標, 一輪車のy座標, サドルの前方向の回転した角度, サドルの横方向の回転した角度, サドルのz座標
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() call will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debuggin, and sometimes learning)
        """

        # 左足の力
        p.applyExternalForce(
            self.unicycle_id,
            self.joints_id["joint_left_pedal"],
            [0, action[0], 0],
            [0, 0, 0],
            flags=p.LINK_FRAME,
        )
        # 右足の力
        p.applyExternalForce(
            self.unicycle_id,
            self.joints_id["joint_right_pedal"],
            [0, action[1], 0],
            [0, 0, 0],
            flags=p.LINK_FRAME,
        )
        # サドルにかける力
        p.applyExternalForce(
            self.unicycle_id,
            self.joints_id["joint_saddle"],
            [action[2], 0, action[3]],
            [0, 0, 0],
            flags=p.LINK_FRAME,
        )

        p.stepSimulation()
        self._update_coordinate()

        return self._get_observation(), self._calc_reward(), False, {}

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.
        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.
        Returns:
            observation (object): the initial observation.
        """
        p.resetBasePositionAndOrientation(self.unicycle_id, self.p_start, p.getQuaternionFromEuler([0, 0, 0]))
        self._update_coordinate()
        return self._get_observation()

    def _update_coordinate(self):
        self.p_robot = np.array(p.getBasePositionAndOrientation(self.unicycle_id)[0])
        self.p_saddle = np.array(p.getLinkState(self.unicycle_id, self.joints_id["joint_saddle"])[0])

    def _get_observation(self) -> List[float]:
        orientation: List[float]
        _, orientation, _, _, _, _ = p.getLinkState(
            self.unicycle_id, self.joints_id["joint_saddle"]
        )
        orientation = p.getEulerFromQuaternion(orientation)
        orientation = [int(orn * 100) / 100 for orn in orientation]
        observation = [
            self.p_goal[0] - self.p_robot[0],
            self.p_goal[1] - self.p_robot[1],
            orientation[1],
            orientation[0] - 1.57,
            self.p_saddle[2],
        ]
        return observation

    def _calc_reward(self) -> float:
        position_diff = np.linalg.norm(self.p_goal - self.p_robot) # type: ignore
        return -position_diff + self.p_saddle[2]


def main():
    env = UnicycleEnv()
    for _ in range(100):
        for _ in range(100):
            env.step([100, 100, 0, 0])
            time.sleep(0.01)
        env.reset()


if __name__ == "__main__":
    main()

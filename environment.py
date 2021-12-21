import pybullet as p
import pybullet_data
from typing import List, Tuple, Dict, Optional
import gym
from gym.spaces import Box
import numpy as np
import time


class UnicycleEnv(gym.Env):
    pos_goal: np.ndarray = np.array([10, 5, 0.3])
    pos_start: np.ndarray = np.array([0, 0, 0.3])
    orn_start: np.ndarray = np.ndarray([0, 0, 0])
    pos_robot: np.ndarray = pos_start.copy()
    orn_robot: np.ndarray = orn_start.copy()
    pos_human: np.ndarray = pos_start.copy()

    # トルク = (-1 ~ 1) x torque_scale
    torque_scale = 10
    # 座標 = (-1 ~ 1) x human_scale
    human_scale = 0.1
    metadata = {"render.modes": ["ansi"]}
    action_space = Box(low=-1, high=1, shape=(2,))
    reward_range: Tuple[float, float] = (-100, 1)
    observation_space: Box = Box(low=-20, high=20, shape=(7,))


    def __init__(self, time_step=0.05, time_wait=None, debug=False, visualize=True):
        self.is_debug = debug
        self.time_wait = time_wait

        self.physicsClient: int
        if visualize == True:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setTimeStep(time_step)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        if self.is_debug:
            p.addUserDebugParameter("wheel", -1, 1, 0)
            p.addUserDebugParameter("human", -1, 1, 0)

        self._setup()

    def _setup(self):
        p.setGravity(0, 0, -10)

        self.plane_id: int = p.loadURDF("plane.urdf")
        self.unicycle_id: int = p.loadURDF("./model/robot.urdf", self.pos_start)

        self.joints_id: dict = dict()
        for i in range(p.getNumJoints(self.unicycle_id)):
            info = p.getJointInfo(self.unicycle_id, i)
            joint_name = info[1].decode("utf-8")
            self.joints_id[joint_name] = i


    def step(self, action: List):
        """Run one timestep of the environment's dynamics.
        When end of episode is reached, you are responsible for calling `reset()` to reset this environment's state

        Args:
            action (object): [車輪のトルク, 人の位置]

        Returns:
            Observation (object): [目標-一輪車のx座標, 目標-一輪車のy座標, 一輪車のz座標] + [一輪車の傾き(四元数)]
            reward (float): 報酬
            done: 終了したかどうか
            info (dict): デバッグ用の情報
        """

        if self.is_debug:
            self._apply_wheel_torque(p.readUserDebugParameter(0))
            self._apply_human(p.readUserDebugParameter(1))
        else:
            self._apply_wheel_torque(action[0])
            self._apply_human(action[1])

        p.stepSimulation()
        self._update_coordinate()

        if self.time_wait is not None:
            time.sleep(self.time_wait)

        return self._get_observation(), self._calc_reward(), self._decide_is_end(), self._get_info()

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
        #p.resetSimulation()
        #self._setup()
        p.resetBasePositionAndOrientation(
            self.unicycle_id, self.pos_start, p.getQuaternionFromEuler([0, 0, 0])
        )
        self._update_coordinate()
        p.resetJointState(self.unicycle_id, self.joints_id["wheel"], 0, 0)
        p.resetJointState(self.unicycle_id, self.joints_id["human"], 0, 0)
        return self._get_observation()

    def _update_coordinate(self):
        position_and_orientation = p.getBasePositionAndOrientation(self.unicycle_id)
        self.pos_robot = np.array(position_and_orientation[0])
        self.orn_robot = np.array(p.getEulerFromQuaternion(position_and_orientation[1]))
        self.pos_human = np.array(
            p.getLinkState(self.unicycle_id, self.joints_id["human"])[0]
        )

    def _get_observation(self) -> List[float]:
        orientation: List[float]
        _, orientation, _, _, _, _ = p.getLinkState(
            self.unicycle_id, self.joints_id["human"]
        )
        observation = [
            self.pos_goal[0] - self.pos_robot[0],
            self.pos_goal[1] - self.pos_robot[1],
            self.pos_robot[2],
        ] + list(orientation)
        return observation

    def _calc_reward(self) -> float:
        #return self.pos_human[2]
        position_diff = np.linalg.norm(self.pos_goal - self.pos_robot)  # type: ignore
        return -position_diff + self.pos_human[2] * 10


    def _decide_is_end(self):
        return self.pos_robot[2] < 0.2

    def _get_info(self) -> Dict[str, np.ndarray]:
        return {}
        return {"pos_robot": self.pos_robot, "orn_robot": self.orn_robot}

    def _apply_wheel_torque(self, torque: float):
        torque = max(torque, -1)
        torque = min(torque, 1)

        p.applyExternalTorque(
            self.unicycle_id,
            self.joints_id["wheel"],
            [0, 0, torque * self.torque_scale],
            flags=p.LINK_FRAME,
        )

    def _apply_human(self, position: float):
        position = max(position, -1)
        position = min(position, 1)
        p.setJointMotorControl2(
            self.unicycle_id,
            self.joints_id["human"],
            p.POSITION_CONTROL,
            position * self.human_scale,
        )


def main():
    env = UnicycleEnv(time_step=0.01, debug=True)
    for _ in range(100):
        for i in range(5000):
            _, reward, is_done, info = env.step([0, 1])
            print(reward)
            if is_done:
                print('done')
                break
            time.sleep(0.01)
        print('reset')
        env.reset()
        #for _ in range(1000):
        #    env.step([0, 0])
        #    time.sleep(0.01)


if __name__ == "__main__":
    main()

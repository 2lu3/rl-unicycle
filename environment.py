import pybullet as p
import pybullet_data
from typing import List, Tuple, Dict
import gym
from gym.spaces import Box, Discrete
import numpy as np
import time
from PIL import Image


class UnicycleEnv(gym.Env):
    # 定数
    time_step = 0.01  # 1ステップの時間
    torque_scale = 5  # トルク = (-1 ~ 1) x torque_scale
    human_scale = 0.05  # 座標 = (-1 ~ 1) x human_scale

    pos_goal: np.ndarray = np.array([5, 0, 0.3])

    #pos_start: np.ndarray = np.concatenate([np.random.random(2) *10, np.array([0.3])])
    pos_start: np.ndarray = np.array([0, 0, 0.3])
    orn_start: np.ndarray = np.array([0, 0, 0])

    pos_previous_robot: np.ndarray = pos_start.copy()

    pos_robot: np.ndarray = pos_start.copy()
    orn_robot: np.ndarray = orn_start.copy()

    pos_human: float = 0

    metadata = {"render.modes": ["ansi"]}
    # action_space = Box(low=-1, high=1, shape=(2,))
    action_space = Discrete(2)
    reward_range: Tuple[float, float] = (-1, 1)
    #observation_space: Box = Box(low=-20, high=20, shape=(16,))
    observation_space: Box = Box(low=-20, high=20, shape=(7,))
    #observation_space: Box = Box(low=-20, high=20, shape=(2,))

    recording = False
    video: List = []

    step_id = 0

    failed_num = 0

    def __init__(self, time_wait=None, debug=False, visualize=True):
        def _connect_physic_client(visualize):
            if visualize == True:
                return p.connect(p.GUI)
            else:
                return p.connect(p.DIRECT)

        self.is_debug = debug
        self.time_wait = time_wait

        self.physicsClient: int = _connect_physic_client(visualize)
        p.setTimeStep(self.time_step)
        p.setGravity(0, 0, -10)

        self._load_urdf()
        self._load_joints()

        if self.is_debug:
            p.addUserDebugParameter("wheel", -1, 1, 0)
            p.addUserDebugParameter("human", -1, 1, 0)

        with open("action.log", "w") as _:
            pass

    def step(self, action: int):
        """Run one timestep of the environment's dynamics.
        When end of episode is reached, you are responsible for calling `reset()` to reset this environment's state

        Args:
            action (object): 行動

        Returns:
            Observation (object): 状態変数
            reward (float): 報酬
            done: 終了したかどうか
            info (dict): デバッグ用の情報
        """

        with open("action.log", "a") as f:
            f.write(str(action) + ' ')
        if self.is_debug:
            self._apply_wheel_torque(p.readUserDebugParameter(0))
            self._apply_human(p.readUserDebugParameter(1))
        else:
            # self._apply_wheel_torque(action[0])
            # self._apply_human(action[1])
            self._apply_wheel_torque(0)
            if action == 0:
                self._apply_human(-self.human_scale)
            #elif action == 1:
            #    self._apply_human(0)
            else:
                self._apply_human(self.human_scale)


        p.stepSimulation()
        self.step_id += 1
        self._update_coordinate()

        if self.recording:
            camera_img = p.getCameraImage(320, 320)
            self.video.append(Image.fromarray(camera_img[2]))

        if self.time_wait is not None:
            time.sleep(self.time_wait)

        return (
            self._get_observation(),
            self._calc_reward(),
            self._decide_is_end(),
            self._get_info(),
        )

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
        # p.resetSimulation()
        self.step_id = 0
        #self.pos_start = np.concatenate([np.random.rand(2) * 10, np.array([0.3])])
        self.pos_start = np.array([0, 0, 0.3])
        p.resetBasePositionAndOrientation(
            self.unicycle_id, self.pos_start, p.getQuaternionFromEuler([0, 0, 0])
        )
        p.resetJointState(self.unicycle_id, self.joints_id["wheel"], 0, 0)
        p.resetJointState(self.unicycle_id, self.joints_id["human"], 0, 0)
        self._update_coordinate()
        return self._get_observation()

    def start_recording(self):
        camera_img = p.getCameraImage(320, 320)
        self.video = [Image.fromarray(camera_img[2])]
        self.recording = True



    def save_recording(self, filepath, time_scale):
        self.recording = False
        self.video[0].save(
            filepath,
            save_all=True,
            append_images=self.video[1:],
            duration=self.time_step * 1000 * time_scale,
            loop=1,
        )

    def _load_urdf(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id: int = p.loadURDF("plane.urdf")
        self.unicycle_id: int = p.loadURDF("./model/robot.urdf", self.pos_start)

    def _load_joints(self):
        self.joints_id: dict = dict()
        for i in range(p.getNumJoints(self.unicycle_id)):
            info = p.getJointInfo(self.unicycle_id, i)
            joint_name = info[1].decode("utf-8")
            self.joints_id[joint_name] = i

    def _update_coordinate(self):
        position_and_orientation = p.getBasePositionAndOrientation(self.unicycle_id)
        self.pos_previous_robot = self.pos_robot.copy()
        self.pos_robot = np.array(position_and_orientation[0])
        self.orn_robot = np.array(position_and_orientation[1])
        self.pos_human = p.getJointState(self.unicycle_id, self.joints_id["human"])[0]

    def _get_observation(self) -> List[float]:
        # wheel_velocity = p.getJointState(self.unicycle_id, self.joints_id["wheel"])[1]
        linear_velocity, angular_velocity = p.getBaseVelocity(self.unicycle_id)
        observation = (
            # [
            #    self.pos_goal[0] - self.pos_robot[0],
            #    self.pos_goal[1] - self.pos_robot[1],
            #    self.pos_goal[2] - self.pos_robot[2],
            # ]
            #list(self.pos_robot)
            list(self.orn_robot)
            + list(p.getEulerFromQuaternion(self.orn_robot))
            #+ list(linear_velocity)
            #+ list(angular_velocity)
            #+ [self.pos_human, wheel_velocity]
        )
        #observation = [self.pos_robot[1], linear_velocity[1]]
        return observation

    def _calc_reward(self) -> float:
        if self.pos_robot[2] < 0.2:
            return -1
        else:
            return 1

        previous_diff = np.linalg.norm(self.pos_goal - self.pos_previous_robot)  # type: ignore
        position_diff = np.linalg.norm(self.pos_goal - self.pos_robot)  # type: ignore
        # initial_diff = np.linalg.norm(self.pos_goal - self.pos_start)  # type: ignore

        if previous_diff < position_diff:
            return -1
        if self.pos_robot[2] < 0.15:
            return -1
        else:
            return 1

    def _decide_is_end(self):
        if self.pos_robot[2] < 0.05:
            self.failed_num += 1
            if self.failed_num > 10:
                self.failed_num = 0
                return True
            else:
                self.reset()
        return False
        return self.pos_robot[2] < 0.05

    def _get_info(self) -> Dict[str, np.ndarray]:
        return {}

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
    env = UnicycleEnv(debug=False)
    for _ in range(100):
        for _ in range(500):
            _, _, is_done, _ = env.step(0)
            if is_done:
                print("done")
                break
            time.sleep(0.001)
        print("reset")
        env.reset()


if __name__ == "__main__":
    main()

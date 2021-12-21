import environment
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Concatenate
from rl.callbacks import ModelIntervalCheckpoint
from tensorflow.keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


class Runner:
    window_length = 1

    def __init__(self, use_gui=True, time_wait=None, record=False):
        self.env = environment.UnicycleEnv(
            visualize=use_gui, time_wait=time_wait, record=record
        )
        self.observation_shape = (self.window_length,) + self.env.observation_space.shape  # type: ignore
        self.nb_actions: int = self.env.action_space.shape[0]  # type: ignore

    def _get_actor_model(self):
        neuron_num = 100
        input_layer = Input(shape=self.observation_shape)  # type: ignore
        c = Flatten()(input_layer)
        c = Dense(neuron_num, activation="tanh")(c)
        for _ in range(6):
            c = Dense(neuron_num, activation="relu")(c)
        c = Dense(self.nb_actions, activation="tanh")(c)  # 出力は連続値なので、linearを使う
        model = Model(input_layer, c)
        print(model.summary())
        return model

    def _get_critic_model(self):
        neuron_num = 200
        action_input = Input(shape=(self.nb_actions,), name="action_input")
        observation_input = Input(
            shape=self.observation_shape, name="observation_input"
        )
        flattened_observation = Flatten()(observation_input)

        input_laer = Concatenate()([action_input, flattened_observation])
        c = Dense(neuron_num, activation="swish")(input_laer)
        for _ in range(6):
            c = Dense(neuron_num, activation="relu")(c)
        c = Dense(1, activation="linear")(c)
        critic = Model(inputs=[action_input, observation_input], outputs=c)
        print(critic.summary())
        return critic, action_input

    def prepare(self):
        # Experience Bufferの設定
        memory = SequentialMemory(limit=50000, window_length=self.window_length)
        # 行動選択時に加えるノイズ(探索のため)
        # 平均回帰課程を用いている。単純にノイズがmuに収束している
        # ここでは、mu=0に設定しているので、ノイズは0になっていく。つまり、探索を行わなくなる。
        random_process = OrnsteinUhlenbeckProcess(
            size=self.nb_actions, theta=0.15, mu=0.1, sigma=0.3
        )

        actor = self._get_actor_model()
        critic, action_input = self._get_critic_model()

        self.agent = DDPGAgent(
            nb_actions=self.nb_actions,
            actor=actor,
            critic=critic,
            critic_action_input=action_input,
            memory=memory,
            nb_steps_warmup_actor=100,
            nb_steps_warmup_critic=100,
            random_process=random_process,
            gamma=0.99,
            target_model_update=1e-3,
        )

        # agent.compile(Adam(lr=.0001, clipnorm=1.), metrics=['mae'])
        self.agent.compile(Adam())

    def learn(self, folder_name: str, step_per_episode: int, nb_steps: int):
        if folder_name[-1] != os.sep:
            folder_name = folder_name + os.sep
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        print("saving to", folder_name)

        callback = ModelIntervalCheckpoint(
            folder_name + "checkpoint_{step:010d}_{reward:2.2f}.h5",
            interval=1000,
            verbose=0,
        )

        histories = self.agent.fit(
            self.env,
            nb_steps=nb_steps,
            visualize=False,
            verbose=1,
            nb_max_episode_steps=step_per_episode,
            callbacks=[callback],
        )

        self.agent.save_weights(folder_name + "ddpg_weights", overwrite=True)

        histories = histories.history
        print(histories)
        plt.plot(
            np.arange(len(histories["episode_reward"])), histories["episode_reward"]
        )
        plt.xlabel("step")
        plt.ylabel("reward")
        plt.savefig(folder_name + "result.png")
        plt.show()

        with open(folder_name + "result.txt", "w") as f:
            for i in range(len(histories["episode_reward"])):
                string = f'{histories["nb_steps"][i]} {histories["episode_reward"][i]}'
                f.write(string + "\n")
                print(string)

    def trial(self):
        result = self.agent.test(self.env, nb_episodes=1, visualize=False)
        print(result)

    def save_gif(self, gif_path):
        if gif_path[-1] != os.sep:
            gif_path = gif_path + os.sep
        if not os.path.exists(gif_path):
            os.mkdir(gif_path)
        print("saving to", gif_path + "result.gif")
        self.env.save_img(gif_path + "result.gif")

    def load_weights(self, filepath):
        self.agent.load_weights(filepath)


class Tester:
    def __init__(self):
        self.env = environment.UnicycleEnv()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--learn", help="重みなどを保存するフォルダ名")
    parser.add_argument(
        "--step-per-episode", help="エピソードあたりのステップ", default=500, type=int
    )
    parser.add_argument("--nb-steps", help="総ステップ数", default=100000, type=int)
    parser.add_argument("--no-gui", help="GUIを表示するか", default=False)
    parser.add_argument("--trial", help="重みをロードするファイル名")
    parser.add_argument("--save-gif", help="gifファイルを保存するフォルダ名", default=None)

    args = parser.parse_args()

    no_gui = args.no_gui == False
    record = args.save_gif is not None
    if args.trial is None:
        runner = Runner(use_gui=no_gui)
        runner.prepare()
        runner.learn(args.learn, args.step_per_episode, args.nb_steps)
    else:
        runner = Runner(time_wait=0.1, use_gui=no_gui, record=record)
        runner.prepare()
        runner.load_weights(args.trial)
        runner.trial()
        if record == True:
            runner.save_gif(args.save_gif)


if __name__ == "__main__":
    main()

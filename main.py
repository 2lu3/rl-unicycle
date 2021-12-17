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
import datetime
import time


class Runner:
    window_length = 1

    def __init__(self):
        self.env = environment.UnicycleEnv()

        self.observation_shape = (self.window_length,) + self.env.observation_space.shape  # type: ignore
        self.nb_actions: int = self.env.action_space.shape[0]  # type: ignore

        self.model = self._get_actor_model()

    def _get_actor_model(self):
        neuron_num = 8
        input_layer = Input(shape=self.observation_shape)  # type: ignore
        c = Flatten()(input_layer)
        c = Dense(neuron_num, activation="swish")(c)
        c = Dense(neuron_num, activation="swish")(c)
        c = Dense(neuron_num, activation="relu")(c)
        c = Dense(neuron_num, activation="relu")(c)
        c = Dense(self.nb_actions, activation="tanh")(c)  # 出力は連続値なので、linearを使う
        model = Model(input_layer, c)
        print(model.summary())
        return model

    def _get_critic_model(self):
        neuron_num = 16
        action_input = Input(shape=(self.nb_actions,), name="action_input")
        observation_input = Input(
            shape=self.observation_shape, name="observation_input"
        )
        flattened_observation = Flatten()(observation_input)

        input_laer = Concatenate()([action_input, flattened_observation])
        c = Dense(neuron_num, activation="swish")(input_laer)
        c = Dense(neuron_num, activation="relu")(c)
        c = Dense(neuron_num, activation="relu")(c)
        c = Dense(neuron_num, activation="relu")(c)
        c = Dense(neuron_num, activation="relu")(c)
        c = Dense(1, activation="linear")(c)
        critic = Model(inputs=[action_input, observation_input], outputs=c)
        print(critic.summary())
        return critic, action_input

    def run(self):
        # Experience Bufferの設定
        memory = SequentialMemory(limit=50000, window_length=self.window_length)
        # 行動選択時に加えるノイズ(探索のため)
        # 平均回帰課程を用いている。単純にノイズがmuに収束している
        # ここでは、mu=0に設定しているので、ノイズは0になっていく。つまり、探索を行わなくなる。
        random_process = OrnsteinUhlenbeckProcess(
            size=self.nb_actions, theta=0.15, mu=0.0, sigma=0.3
        )

        actor = self._get_actor_model()
        critic, action_input = self._get_critic_model()

        agent = DDPGAgent(
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
        agent.compile(Adam())

        callback = ModelIntervalCheckpoint(
            "weight/checkpoint_{episode:06d}_{reward:2.2f}.h5", interval=10000, verbose=0
        )

        step_per_episode = 500
        all_step_num = step_per_episode * 10000

        history = agent.fit(
            self.env,
            nb_steps=all_step_num,
            visualize=False,
            verbose=1,
            nb_max_episode_steps=step_per_episode,
            callbacks=[callback],
        )

        agent.save_weights("weight/ddpg_weights_{}.h5f".format(datetime.datetime.now()))

        print(history.history)

        history = history.history
        print(history)
        with open("result.txt", "w") as f:
            f.write(history)
        plt.plot(np.arange(len(history["episode_reward"])), history["episode_reward"])
        plt.savefig("result.png")
        plt.show()

        # agent.test(self.env, nb_episodes=5, visualize=True)


class Tester:
    def __init__(self):
        self.env = environment.UnicycleEnv()


def main():
    runner = Runner()
    runner.run()

if __name__ == "__main__":
    main()

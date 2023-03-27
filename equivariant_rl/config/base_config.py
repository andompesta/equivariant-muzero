# taken from https://raw.githubusercontent.com/werner-duvaud/muzero-general/0825bd544fc172a2e2dcc96d43711123222c4a2f/games/abstract_game.py
import gym
from abc import ABC, abstractmethod
from torchrl.envs import ParallelEnv
from torchrl.envs.libs.gym import GymWrapper

from equivariant_rl.env import ProcGymWrapper
from .reanalizer_config import ReanalyzerConfig
from typing import Optional


class BaseConfig(ABC):

    def __init__(
        self,
        env_name: str,
        num_envs: int,

        replay_buffer_size: int = int(1e6),
        num_unroll_steps: int = 5,
        td_steps: int = 10,
        PER: bool = True,
        PER_alpha: float = 1.,

        seed: int = 42,
        self_play_delay: int = 0,
        training_delay: int = 0,
        ratio: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_envs = num_envs
        self.env_name = env_name
        self.reanalyzer_config: Optional[ReanalyzerConfig] = None

        # replay buffer
        self.replay_buffer_size = replay_buffer_size,
        self.num_unroll_steps = num_unroll_steps,
        self.td_steps = td_steps,
        self.PER = PER
        self.PER_alpha = PER_alpha

        # Adjust the self play / training ratio to avoid over/underfitting
        self.seed: int = seed
        self.self_play_delay = self_play_delay  # Number of seconds to wait after each played game
        self.training_delay = training_delay  # Number of seconds to wait after each training step
        self.ratio = ratio  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

        for name, value in kwargs.items():
            if not hasattr(self, name):
                setattr(
                    self,
                    name,
                    value,
                )

    def get_env(self) -> GymWrapper:
        if self.num_envs <= 1:
            return self._get_env()
        else:
            return ParallelEnv(
                num_workers=self.num_envs,
                create_env_fn=self._get_env,
            )

    def _get_env(self) -> GymWrapper:
        return GymWrapper(
            # from gym to torchrl
            ProcGymWrapper(
                # from gym3 to gym
                gym.make(self.env_name).unwrapped
            )
        )

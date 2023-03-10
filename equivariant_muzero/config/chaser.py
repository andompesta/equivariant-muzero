from .replay_buffer_config import ReplayBufferConfig
from .base_config import BaseConfig
from procgen import ProcgenGym3Env


class ChaserConfig(BaseConfig):

    def __init__(
        self,
        num_envs: int = 1,
        replay_buffer_size: int = int(1e6),
        num_unroll_steps: int = 5,
        td_steps: int = 10,
        PRE: bool = True,
        PRE_alpha: float = 1.,
        **kwargs,
    ) -> None:
        """
        Game configuration class

        :param num_envs: number of environmant, defaults to 1
        :type num_envs: int, optional
        :param replay_buffer_size: number of trace to keep in the replay buffer, defaults to int(1e6)
        :type replay_buffer_size: int, optional
        :param num_unroll_steps: number of actions to keep for every element of the batch, defaults to 5
        :type num_unroll_steps: int, optional
        :param td_steps: number of steps in the future to take into account for calculating the target value, defaults to 10
        :type td_steps: int, optional
        :param PRE: enable prioritized replay: prioritize unexpected element in the replay buffer, defaults to True
        :type PRE: bool, optional
        :param PRE_alpha: how much prioritization is used, 0 corresponding to the uniform case, defaults to 1.
        :type PRE_alpha: float, optional
        """
        super().__init__(
            env_name="chaser",
            num_envs=num_envs,
            **kwargs,
        )

        self.replay_buffer_size = replay_buffer_size,
        self.num_unroll_steps = num_unroll_steps,
        self.td_steps = td_steps,
        self.PER = PRE
        self.PER_alpha = PRE_alpha

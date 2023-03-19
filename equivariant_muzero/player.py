import ray
import gym3
import numpy as np
import torch
from procgen import ProcgenEnv

from equivariant_muzero.replay_buffer import ReplayBuffer

@ray.remote
class SelfPlayer:
    def __init__(
        self,
        env: ProcgenEnv,
        seed: int,
        device: str,
    ) -> None:
        self.env = env
        self.seed = seed
        self.device = torch.device(device)

        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def init_replay_buffer(
        self,
        replay_buffer: ReplayBuffer,
        
    ):
        

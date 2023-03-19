from dataclasses import dataclass
from typing import Dict, Optional
from torch import Tensor

@dataclass
class Checkpoint:
    weights: Optional[Dict[str, Tensor]] = None
    optimizer_state: Optional[Dict[str, Tensor]] = None
    total_reward: float = 0.
    muzero_reward: float = 0
    opponent_reward: float = 0
    episode_length: int = 0
    mean_value: float = 0
    training_step: int = 0
    lr: float = 0
    total_loss: float = 0
    value_loss: float = 0
    reward_loss: float = 0
    policy_loss: float = 0
    num_played_games: int = 0
    num_played_steps: int = 0
    num_reanalysed_games: int = 0
    terminate: bool = False

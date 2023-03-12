from dataclasses import dataclass

@dataclass
class ReplayBufferConfig:
    replay_buffer_size: int  # 
    num_unroll_steps: int  # Number of game moves to keep for every batch element
    td_steps: int  # Number of steps in the future to take into account for calculating the target value
    
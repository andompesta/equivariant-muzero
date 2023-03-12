# taken from https://raw.githubusercontent.com/werner-duvaud/muzero-general/0825bd544fc172a2e2dcc96d43711123222c4a2f/games/abstract_game.py
from abc import ABC, abstractmethod
from procgen import ProcgenGym3Env
from .replay_buffer_config import ReplayBufferConfig
from .reanalizer_config import ReanalyzerConfig
from typing import Optional


class BaseConfig(ABC):
    def __init__(
        self,
        env_name: str,
        num_envs: int = 1,
        seed: int = 42,
        self_play_delay: int = 0,
        training_delay: int = 0,
        ratio: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.num_envs = num_envs,
        self.env_name = env_name
        self.reanalyzer_config: Optional[ReanalyzerConfig] = None

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

    def get_env(self) -> ProcgenGym3Env:
        return ProcgenGym3Env(
            num=self.num_envs,
            env_name=self.env_name,
        )

    # @abstractmethod
    # def step(self, action):
    #     """
    #     Apply action to the game.

    #     Args:
    #         action : action of the action_space to take.

    #     Returns:
    #         The new observation, the reward and a boolean if the game has ended.
    #     """
    #     pass

    # def to_play(self):
    #     """
    #     Return the current player.

    #     Returns:
    #         The current player, it should be an element of the players list in the config.
    #     """
    #     return 0

    # @abstractmethod
    # def legal_actions(self):
    #     """
    #     Should return the legal actions at each turn, if it is not available, it can return
    #     the whole action space. At each turn, the game have to be able to handle one of returned actions.

    #     For complex game where calculating legal moves is too long, the idea is to define the legal actions
    #     equal to the action space but to return a negative reward if the action is illegal.

    #     Returns:
    #         An array of integers, subset of the action space.
    #     """
    #     pass

    # @abstractmethod
    # def reset(self):
    #     """
    #     Reset the game for a new game.

    #     Returns:
    #         Initial observation of the game.
    #     """
    #     pass

    # def close(self):
    #     """
    #     Properly close the game.
    #     """
    #     pass

    # @abstractmethod
    # def render(self):
    #     """
    #     Display the game observation.
    #     """
    #     pass

    # def human_to_action(self):
    #     """
    #     For multiplayer games, ask the user for a legal action
    #     and return the corresponding action number.

    #     Returns:
    #         An integer from the action space.
    #     """
    #     choice = input(f"Enter the action to play for the player {self.to_play()}: ")
    #     while int(choice) not in self.legal_actions():
    #         choice = input("Illegal action. Enter another action : ")
    #     return int(choice)

    # def expert_agent(self):
    #     """
    #     Hard coded agent that MuZero faces to assess his progress in multiplayer games.
    #     It doesn't influence training

    #     Returns:
    #         Action as an integer to take in the current game state
    #     """
    #     raise NotImplementedError

    # def action_to_string(self, action_number):
    #     """
    #     Convert an action number to a string representing the action.

    #     Args:
    #         action_number: an integer from the action space.

    #     Returns:
    #         String representing the action.
    #     """
    #     return str(action_number)

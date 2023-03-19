from dataclasses import dataclass, field
from typing import List, Optional
# from torch import Tensor
import numpy as np


@dataclass
class GameHistory:
    observation_hisotry: List[np.ndarray] = field(default_factory=[])
    action_history: List[int] = field(default_factory=[])
    reward_history: List[float] = field(default_factory=[])
    player_history: List[int] = field(default_factory=[])
    child_visits: List[List[int]] = field(default_factory=[])
    root_values: List[Optional[float]] = field(default_factory=[])
    reanalysed_predicted_root_values = None
    # For PER
    priorities: Optional[List[float]] = None
    game_priority: Optional[float] = None

    def store_search_statistics(
        self,
        root,
        action_space: List[int],
    ):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(
                child.visit_count for child in root.children.values())
            self.child_visits.append([
                root.children[a].visit_count /
                sum_visits if a in root.children else 0 for a in action_space
            ])

            self.root_values.append(root.value())
        else:
            self.root_values.append(None)

    def get_stacked_observations(
        self,
        index,
        num_stacked_observations,
        action_space_size,
    ) -> np.ndarray:
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """
        # Convert to positive index
        index = index % len(self.observation_history)

        stacked_observations = self.observation_history[index].copy()
        for past_observation_index in reversed(
                range(index - num_stacked_observations, index)):
            if 0 <= past_observation_index:
                previous_observation = np.concatenate((
                    self.observation_history[past_observation_index],
                    [
                        np.ones_like(stacked_observations[0]) *
                        self.action_history[past_observation_index + 1] /
                        action_space_size
                    ],
                ))
            else:
                previous_observation = np.concatenate((
                    np.zeros_like(self.observation_history[index]),
                    [np.zeros_like(stacked_observations[0])],
                ))

            stacked_observations = np.concatenate(
                (stacked_observations, previous_observation))

        return stacked_observations

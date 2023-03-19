from typing import Optional

from gym import Env, Wrapper
from gym.utils.passive_env_checker import (
    check_action_space,
    check_observation_space,
    env_render_passive_checker,
    env_reset_passive_checker,
)


class ProcGymWrapper(Wrapper):

    def __init__(
        self,
        env: Env,
        **kwargs,
    ):
        super().__init__(
            env,
            **kwargs,
        )


    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Resets the environment that on the first call will run the `passive_env_reset_check`."""
        self.seed = seed
        self.options = options
        return self.env.reset()

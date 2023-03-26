from typing import Optional

from gym import Env, Wrapper


class ProcGymWrapper(Wrapper):

    def __init__(
        self,
        env: Env,
        seed: Optional[int] = 42,
        options: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            env,
            **kwargs,
        )
        self.seed = seed

    def reset(
        self,
        seed: Optional[int] = 42,
        options: Optional[dict] = None,
    ):
        """Resets the environment that on the first call will run the `passive_env_reset_check`."""
        self.seed = seed
        self.options = options
        return self.env.reset()

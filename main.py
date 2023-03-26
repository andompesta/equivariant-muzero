import torch
import wandb
from equivariant_muzero.replay_buffer import ReplayBuffer
from equivariant_muzero.config import ChaserConfig
from equivariant_muzero.utils.images import observation_as_image
from torchrl.envs.utils import check_env_specs

from torchrl.envs import (
    Compose,
    NoopResetEnv,
    ObservationNorm,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
)

import ray

if __name__ == "__main__":
    config = ChaserConfig(
        num_envs=0,
    )
    env = config.get_env()
    check_env_specs(env)

    env = TransformedEnv(
        env,
        Compose(
            StepCounter(),
            ToTensorImage(),
            # ObservationNorm(),
        ),
    )

    state = env.reset()
    observation_as_image(state["pixels"])

    ray.init()
    replay_buffer = ReplayBuffer.remote(
        config=config,
        buffer=dict(),
    )


    rew, obs, is_first = env.observe()
    obs = torch.tensor(obs["rgb"])
    obs = obs.to("cuda")
    print(obs)
    # sto cazzo

    wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        })

    wandb.log({"acc": 1, "loss": 0})

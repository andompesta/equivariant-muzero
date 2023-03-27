import torch
import wandb

from torchrl.envs.utils import check_env_specs


from equivariant_rl.replay_buffer import ReplayBuffer
from equivariant_rl.config import ChaserConfig
from equivariant_rl.env.utils import make_transformed_env


if __name__ == "__main__":
    config = ChaserConfig(
        num_envs=0,
    )
    env = config.get_env()
    check_env_specs(env)
    env = make_transformed_env(env)



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

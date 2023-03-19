import torch
import wandb
from equivariant_muzero.replay_buffer import ReplayBuffer
from equivariant_muzero.config import ChaserConfig
import ray

if __name__ == "__main__":
    config = ChaserConfig(
        num_envs=1,
    )
    env = config.get_env()

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

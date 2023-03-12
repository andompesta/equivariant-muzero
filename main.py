import torch
import wandb
from procgen import ProcgenGym3Env
from equivariant_muzero.config.replay_buffer_config import ReplayBufferConfig
from equivariant_muzero.config.chaser import ChaserConfig

if __name__ == "__main__":
    env = ProcgenGym3Env(
        num=1,
        env_name="chaser",
    )

    config = ChaserConfig()

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

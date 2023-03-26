from matplotlib import pyplot as plt
from torch import Tensor
# matplotlib set style
plt.style.use("seaborn-v0_8-whitegrid")


def observation_as_image(obs: Tensor, title=None):
    plt.imshow(obs.permute(1, 2, 0).cpu().numpy())
    if title is not None:
        plt.title(title)
    plt.show()
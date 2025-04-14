import os
import sys
import warnings

import hydra
from omegaconf import DictConfig

os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(1, ".")

from src.dataset import build_dataset, build_instruct_dataset


@hydra.main(
    config_path="../configs/pretrain", config_name="pretrain_dataset", version_base=None
)
def main(cfg: DictConfig):
    if cfg.type == "pretrain":
        build_dataset(cfg)
    elif cfg.type == "instruct":
        build_instruct_dataset(cfg)
    else:
        raise ValueError(f"Unexpected dataset type: {cfg.type}")


if __name__ == "__main__":
    main()

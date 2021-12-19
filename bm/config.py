import dataclasses
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()


@dataclasses.dataclass
class ProjectPaths:
    data_dir = BASE_DIR.joinpath('data')
    checkpoints_dir = BASE_DIR.joinpath('model_checkpoints')


@dataclasses.dataclass
class Config:
    random_seed = 7462294
    train_size = 0.7
    val_size = 0.15

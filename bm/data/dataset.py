import pandas as pd
from torch.utils.data import Dataset
import itertools
import albumentations as A
from bm.config import ProjectPaths
import pathlib
from cv2 import cv2


class BacteriaCombinationsDataset(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 transforms: A.Compose = None,
                 data_dir: pathlib.Path = ProjectPaths.data_dir):
        # Make combinations of photos from different sequences
        dataframe = dataframe.sort_values(['sequence_id', 'frame_idx']).reset_index(drop=True)
        self.combinations_list = list()
        for sequence_id in dataframe['sequence_id'].unique():
            values = dataframe[dataframe['sequence_id'] == sequence_id][['frame_idx', 'image_fp']].values.tolist()
            sequence_combinations = list(itertools.product(values, repeat=2))
            self.combinations_list.extend(sequence_combinations)

        self.transforms = transforms
        self.data_dir = data_dir

    def __getitem__(self, item):
        (frame_idx_1, image_fp_1), (frame_idx_2, image_fp_2) = self.combinations_list[item]
        image_fp_1 = self.data_dir.joinpath(image_fp_1)
        image_fp_2 = self.data_dir.joinpath(image_fp_2)
        image_1 = cv2.imread(image_fp_1.as_posix())
        image_2 = cv2.imread(image_fp_2.as_posix())

        if self.transforms is not None:
            image_1 = self.transforms(image=image_1)['image']
            image_2 = self.transforms(image=image_2)['image']

        target = int(frame_idx_1 < frame_idx_2)
        return image_1, image_2, target

    def __len__(self):
        return len(self.combinations_list)

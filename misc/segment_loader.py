import os
import math

from typing import List, Tuple

import numpy as np
import pandas as pd
from skimage.io import imread
from tensorflow.keras.utils import Sequence

from misc.constants import CONST
from misc.mask_rle_processing import RLEMaskProcessor


class SegmentationDataLoader(Sequence):
    def __init__(self, image_set: list, mask_set: pd.DataFrame, batch_size: int) -> None:
        self.__image_set = image_set
        self.__mask_set = mask_set
        self.__batch_size = batch_size


    def __len__(self) -> int:
        return math.ceil(len(self.__image_set) / self.__batch_size)
    

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_images = self.__image_set[index * self.__batch_size 
                                        : (index + 1) * self.__batch_size]
        
        batch_x = np.array([
            imread(os.path.join(CONST.TRAIN_FOLDER, img_name)) 
            for img_name in batch_images
        ])

        batch_y = np.array([
            RLEMaskProcessor.masks_as_image(
                self.__mask_set[self.__mask_set['ImageId'] == img_name]['EncodedPixels']) 
            for img_name in batch_images
        ])

        return batch_x, batch_y

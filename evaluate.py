import os
import numpy as np
import pandas as pd

from keras import models
from skimage.io import imread

from misc.constants import CONST
from misc.mask_rle_processing import RLEMaskProcessor


def main():
    model = models.load_model(CONST.MODEL_FOLDER, compile=False)

    test_paths = os.listdir(CONST.TEST_FOLDER)

    out_pred_rows = [
    {'ImageId': img_id, 'EncodedPixels': encoding if encoding else None}
    for img_id in test_paths
    for encoding in RLEMaskProcessor.multi_rle_encode(model.predict(
        np.expand_dims(
            imread(os.path.join(CONST.TEST_FOLDER, img_id)) / 255.0, 0))[0]) or [None]
    ]

    result_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
    result_df.to_csv('result.csv', index=False)
   
    
if __name__ == "__main__":
    main()
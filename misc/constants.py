import os


class CONST:
    """
    Configuration constants for the Airbus Ship Detection Challenge.
    """
    __slots__=()

    # Paths to training and test data directories
    MODEL_FOLDER = os.path.abspath("../models/model.h5")
    TRAIN_FOLDER = os.path.abspath("../train_v2")
    TEST_FOLDER = os.path.abspath("../test_v2")
    CSV_PATH = os.path.abspath("../train_ship_segmentations_v2.csv")

    # Image configuration
    IMG_SIZE = (768, 768, 3)
    NUM_CLASSES = 2

    # Training configuration
    BATCH_SIZE = 8
    MAX_TRAIN_STEPS = 100
    VALIDATION_SPLIT = 0.2
    NB_EPOCHS = 5
    RANDOM_STATE = 42
    VAL_IMAGES = 500

CONST = CONST()

__all__ = CONST

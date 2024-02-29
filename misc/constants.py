import os


class CONST:
    """
    Configuration constants for the Airbus Ship Detection Challenge.
    """
    __slots__=()

    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    # Paths to training and test data directories
    TEST_FOLDER = os.path.join(DATA_PATH, "test_v2")
    TRAIN_FOLDER = os.path.join(DATA_PATH, "train_v2")
    MODEL_FOLDER = os.path.join(DATA_PATH, "models", "model.h5")
    CSV_PATH = os.path.join(DATA_PATH, "train_ship_segmentations_v2.csv")

    # Image configuration
    IMG_SIZE = (768, 768, 3)
    NUM_CLASSES = 2

    # Training configuration
    BATCH_SIZE = 4
    INITIAL_LEARNING_RATE = 1e-3
    MAX_TRAIN_STEPS = 100
    VALIDATION_SPLIT = 0.2
    NB_EPOCHS = 5
    RANDOM_STATE = 42
    VAL_IMAGES = 500

CONST = CONST()

__all__ = CONST
print(CONST.TRAIN_FOLDER)

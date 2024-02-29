from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow import Tensor


class SegmentationMetrics:
    @staticmethod
    def dice_coefficient(y_true: Tensor, y_pred: Tensor, smooth: float = 1) -> Tensor:
        """
        Calculate the Dice Coefficient for measuring the similarity between two sets of data.

        Args:
            y_true (Tensor): The ground truth binary mask.
            y_pred (Tensor): The predicted binary mask.
            smooth (float): A smoothing factor to avoid division by zero.

        Returns:
            Tensor: The Dice Coefficient as a float between 0 and 1.
        """
        y_true_f = K.cast(y_true, 'float32')
        y_pred_f = K.cast(y_pred, 'float32')  # Ensure y_pred is also cast to float32
        intersection = K.sum(y_true_f * y_pred_f, axis=[1, 2, 3])
        union = K.sum(y_true_f, axis=[1, 2, 3]) + K.sum(y_pred_f, axis=[1, 2, 3])
        dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
        return dice


    @staticmethod
    def combined_dice_bce_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Combine Dice loss and Binary Cross-Entropy (BCE) for segmentations tasks.

        Args:
            y_true (Tensor): The ground truth binary mask.
            y_pred (Tensor): The predicted binary mask.

        Returns:
            Tensor: The combined Dice and BCE loss.
        """
        bce_loss = binary_crossentropy(y_true, y_pred)
        dice_loss = 1 - SegmentationMetrics.dice_coefficient(y_true, y_pred)
        # Ensure the combined loss is always non-negative
        combined_loss = bce_loss + dice_loss
        return combined_loss


    @staticmethod
    def true_positive_rate(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Calculate the True Positive Rate (Sensitivity or Recall) for the predictions.

        Args:
            y_true (Tensor): The ground truth binary mask.
            y_pred (Tensor): The predicted binary mask.

        Returns:
            Tensor: The True Positive Rate as a float between 0 and 1.
        """
        true_positives = K.sum(K.flatten(y_true) * K.flatten(K.round(y_pred)))
        total_true = K.sum(y_true)
        return true_positives / (total_true + K.epsilon())


    @staticmethod
    def precision_metric(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Calculate the Precision of the predictions (Proportion of true positives among all positives).

        Args:
            y_true (Tensor): The ground truth binary mask.
            y_pred (Tensor): The predicted binary mask.

        Returns:
            Tensor: The Precision as a float between 0 and 1.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + K.epsilon())


    @staticmethod
    def recall_metric(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Calculate the Recall (True Positive Rate) of the predictions.

        Args:
            y_true (Tensor): The ground truth binary mask.
            y_pred (Tensor): The predicted binary mask.

        Returns:
            Tensor: The Recall as a float between 0 and 1.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        total_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (total_positives + K.epsilon())


    @staticmethod
    def specificity_metric(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Calculate the Specificity (True Negative Rate) of the predictions.

        Args:
            y_true (Tensor): The ground truth binary mask.
            y_pred (Tensor): The predicted binary mask.

        Returns:
            Tensor: The Specificity as a float between 0 and 1.
        """
        true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        total_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
        return true_negatives / (total_negatives + K.epsilon())


    @staticmethod
    def f1_score_metric(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Calculate the F1 Score, the harmonic mean of Precision and Recall.

        Args:
            y_true (Tensor): The ground truth binary mask.
            y_pred (Tensor): The predicted binary mask.

        Returns:
            Tensor: The F1 Score as a float between 0 and 1.
        """
        precision = SegmentationMetrics.precision_metric(y_true, y_pred)
        recall = SegmentationMetrics.recall_metric(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from misc.constants import CONST
from misc.unet import UnetModel
from misc.data_processing import DataPreparation
from misc.metrics import SegmentationMetrics



def main():
    lr_schedule = ExponentialDecay(
        CONST.INITIAL_LEARNING_RATE,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    
    model: Model = UnetModel(CONST.IMG_SIZE).get_model() # create the U-Net model
    
    # Prepare data
    train_data, valid_data = DataPreparation(CONST.CSV_PATH, CONST.BATCH_SIZE).split_data()
    

    # Compile model with adam optimizer, binary cross-entropy and SegmentationMetrics
    model.compile(optimizer=Adam(learning_rate=lr_schedule),
                loss=lambda y_true, y_pred: SegmentationMetrics.combined_dice_bce_loss(y_true, y_pred),   
                metrics=[SegmentationMetrics.dice_coefficient,
                           "binary_accuracy",
                           SegmentationMetrics.true_positive_rate,
                           SegmentationMetrics.precision_metric,
                           SegmentationMetrics.recall_metric,
                           SegmentationMetrics.specificity_metric,
                           SegmentationMetrics.f1_score_metric])
    
    weight_path = f"models/seg_model_weights.best.hdf5" # Weight path

    # Save the model after each epoch if the validation loss improved
    checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only = True)
    
    # Decrease the learning rate once the metric ceases to show improvement
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, 
                                    patience=3, 
                                    verbose=1, mode='max', min_delta=0.0001, cooldown=2, min_lr=1e-6)

    # Halt the training process once there's no further improvement in validation loss
    early = EarlyStopping(monitor="val_dice_coef", 
                        mode="max", 
                        patience=15)

    # Maintain a record of training progress by assembling a list of callbacks
    callbacks_list = [checkpoint, early, reduceLROnPlat]

    model.fit(train_data,
                validation_data=valid_data, 
                epochs=CONST.NB_EPOCHS, 
                callbacks=callbacks_list)

    model.load_weights(weight_path)
    model.save('models/model.h5')


if __name__ == "__main__":
    main()
from typing import Tuple
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

class UnetModel:
    def __init__(self, input_shape: Tuple[int, int, int]) -> None:
        self.__input_shape = input_shape
        self.__model = self.__build_model()


    def __build_model(self) -> Model:
        # Input layer
        inputs = Input(self.__input_shape)

        # Contracting path
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)

        # Expanding path
        up5 = UpSampling2D(size=(2, 2))(conv4)
        up5 = Conv2D(256, 2, activation='relu', padding='same')(up5)
        merge5 = concatenate([conv3, up5], axis=3)
        conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
        conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

        up6 = UpSampling2D(size=(2, 2))(conv5)
        up6 = Conv2D(128, 2, activation='relu', padding='same')(up6)
        merge6 = concatenate([conv2, up6], axis=3)
        conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

        up7 = UpSampling2D(size=(2, 2))(conv6)
        up7 = Conv2D(64, 2, activation='relu', padding='same')(up7)
        merge7 = concatenate([conv1, up7], axis=3)
        conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

        # Output layer
        output = Conv2D(1, 1, activation='sigmoid')(conv7)

        # Create the model
        model = Model(inputs=inputs, outputs=output)

        return model
    
    
    def get_model(self) -> Model:
        return self.__model
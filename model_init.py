from keras.models import Model, Sequential
from keras.layers import Conv2D, Concatenate, Conv2DTranspose, Input, BatchNormalization, MaxPooling2D, Reshape, Flatten
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.preprocessing import image
from dataset import data_preparation

def autoencoder():

    img_input = Input((28, 28, 1))

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(img_input)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(img_input, x)
    model.compile(optimizer=Adam(), loss='mean_squared_error')

    return model



def train_model(x_train, model):

    model.fit(x_train[:1000], x_train[:1000], epochs=20, batch_size = 100, validation_data = (x_train[1000:2000], x_train[1000:2000]))

    pred = model.predict(x_train[:1000])
    pred *= 255
    pred = pred.astype('uint32')

    return pred
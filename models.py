from tensorflow import keras


# UNet with no skips
def get_model_0(input_shape, output_channel_count):
    model_input = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(model_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)              # 240
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)              # 120
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)              # 60
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)              # 30
    x = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)              # 15

    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 30
    x = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 60
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 120
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 240
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 480
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(output_channel_count, (1, 1), padding='same', activation=None)(x)

    model = keras.Model(inputs=model_input, outputs=x)
    return model


# My Unet
def get_model_1(input_shape, output_channel_count):
    model_input = keras.Input(shape=input_shape)

    conv1 = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(model_input)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(conv1)
    conv1 = keras.layers.BatchNormalization()(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)              # 240
    conv2 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(pool1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(conv2)
    conv2 = keras.layers.BatchNormalization()(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)              # 120
    conv3 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(pool2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv3)
    conv3 = keras.layers.BatchNormalization()(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)              # 60
    conv4 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(pool3)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv4)
    conv4 = keras.layers.BatchNormalization()(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)              # 30
    conv5 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(pool4)
    conv5 = keras.layers.BatchNormalization()(conv5)
    conv5 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(conv5)
    conv5 = keras.layers.BatchNormalization()(conv5)
    pool5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv5)              # 15

    x = keras.layers.UpSampling2D(interpolation='bilinear')(pool5)      # 30
    x = keras.layers.concatenate([conv5, x], axis=3)
    x = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 60
    x = keras.layers.concatenate([conv4, x], axis=3)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 120
    x = keras.layers.concatenate([conv3, x], axis=3)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 240
    x = keras.layers.concatenate([conv2, x], axis=3)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 480
    x = keras.layers.concatenate([conv1, x], axis=3)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(output_channel_count, (1, 1), padding='same', activation=None)(x)

    model = keras.Model(inputs=model_input, outputs=x)
    return model


# Light model
def get_model_2(input_shape, output_channel_count):
    model_input = keras.Input(shape=input_shape)

    conv1 = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(model_input)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(conv1)
    conv1 = keras.layers.BatchNormalization()(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)              # 240
    conv2 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(pool1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(conv2)
    conv2 = keras.layers.BatchNormalization()(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)              # 120
    conv3 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(pool2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv3)
    conv3 = keras.layers.BatchNormalization()(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)              # 60
    conv4 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(pool3)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv4)
    conv4 = keras.layers.BatchNormalization()(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)              # 30

    x = keras.layers.UpSampling2D(interpolation='bilinear')(pool4)      # 60
    x = keras.layers.concatenate([conv4, x], axis=3)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 120
    x = keras.layers.concatenate([conv3, x], axis=3)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 240
    x = keras.layers.concatenate([conv2, x], axis=3)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 480
    x = keras.layers.concatenate([conv1, x], axis=3)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(output_channel_count, (1, 1), padding='same', activation=None)(x)

    model = keras.Model(inputs=model_input, outputs=x)
    return model


def get_hourglass_block(stack_input):
    conv1 = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(stack_input)
    conv1 = keras.layers.BatchNormalization()(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)  # 240
    conv2 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(pool1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)  # 120
    conv3 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(pool2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)  # 60
    conv4 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(pool3)
    conv4 = keras.layers.BatchNormalization()(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)  # 30

    x = keras.layers.UpSampling2D(interpolation='bilinear')(pool4)  # 60
    x = keras.layers.concatenate([conv4, x], axis=3)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)  # 120
    x = keras.layers.concatenate([conv3, x], axis=3)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)  # 240
    x = keras.layers.concatenate([conv2, x], axis=3)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)  # 480
    x = keras.layers.concatenate([conv1, x], axis=3)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    return x


def get_hourglass_model(input_shape, output_channel_count, stack_size=2):
    model_input = keras.Input(shape=input_shape)
    prev_res = get_hourglass_block(model_input)
    x = None
    for index in range(stack_size - 1):
        if x is not None:
            prev_res = keras.layers.concatenate([prev_res, x], axis=3)
        x = get_hourglass_block(prev_res)

    output = keras.layers.Conv2D(output_channel_count, (1, 1), padding='same', activation=None)(x)
    model = keras.Model(inputs=model_input, outputs=output)
    return model


def unet(input_shape, output_channel_count):
    model_input = keras.Input(input_shape)

    conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(model_input)
    conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = keras.layers.Dropout(0.5)(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = keras.layers.Dropout(0.5)(conv5)

    up6 = keras.layers.Conv2D(512, 2, activation='relu', padding='same')(
        keras.layers.UpSampling2D(size=(2, 2))(drop5)
    )
    merge6 = keras.layers.concatenate([drop4, up6], axis=3)
    conv6 = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = keras.layers.Conv2D(256, 2, activation='relu', padding='same')(
        keras.layers.UpSampling2D(size=(2, 2))(conv6)
    )
    merge7 = keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = keras.layers.Conv2D(128, 2, activation='relu', padding='same')(
        keras.layers.UpSampling2D(size=(2, 2))(conv7)
    )
    merge8 = keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = keras.layers.Conv2D(64, 2, activation='relu', padding='same')(
        keras.layers.UpSampling2D(size=(2, 2))(conv8)
    )
    merge9 = keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv10 = keras.layers.Conv2D(output_channel_count, 1, activation=None, padding='same')(conv9)

    model = keras.Model(inputs=model_input, outputs=conv10)
    return model


# My bigger Unet
def get_model_8(input_shape, output_channel_count):
    model_input = keras.Input(shape=input_shape)

    conv1 = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(model_input)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(conv1)
    conv1 = keras.layers.BatchNormalization()(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)              # 240
    conv2 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(pool1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(conv2)
    conv2 = keras.layers.BatchNormalization()(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)              # 120
    conv3 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(pool2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv3)
    conv3 = keras.layers.BatchNormalization()(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)              # 60
    conv4 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(pool3)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv4)
    conv4 = keras.layers.BatchNormalization()(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)              # 30
    conv5 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(pool4)
    conv5 = keras.layers.BatchNormalization()(conv5)
    conv5 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(conv5)
    conv5 = keras.layers.BatchNormalization()(conv5)
    pool5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv5)              # 15

    x = keras.layers.UpSampling2D(interpolation='bilinear')(pool5)      # 30
    x = keras.layers.concatenate([conv5, x], axis=3)
    x = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 60
    x = keras.layers.concatenate([conv4, x], axis=3)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 120
    x = keras.layers.concatenate([conv3, x], axis=3)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 240
    x = keras.layers.concatenate([conv2, x], axis=3)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 480
    x = keras.layers.concatenate([conv1, x], axis=3)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(output_channel_count, (1, 1), padding='same', activation=None)(x)

    model = keras.Model(inputs=model_input, outputs=x)
    return model


# My bigger Unet with l2
def get_model_9(input_shape, output_channel_count):
    model_input = keras.Input(shape=input_shape)

    conv1 = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(model_input)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(conv1)
    conv1 = keras.layers.BatchNormalization()(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)              # 240
    conv2 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(pool1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(conv2)
    conv2 = keras.layers.BatchNormalization()(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)              # 120
    conv3 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(pool2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(conv3)
    conv3 = keras.layers.BatchNormalization()(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)              # 60
    conv4 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(pool3)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(conv4)
    conv4 = keras.layers.BatchNormalization()(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)              # 30
    conv5 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(pool4)
    conv5 = keras.layers.BatchNormalization()(conv5)
    conv5 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(conv5)
    conv5 = keras.layers.BatchNormalization()(conv5)
    pool5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv5)              # 15

    x = keras.layers.UpSampling2D(interpolation='bilinear')(pool5)      # 30
    x = keras.layers.concatenate([conv5, x], axis=3)
    x = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 60
    x = keras.layers.concatenate([conv4, x], axis=3)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 120
    x = keras.layers.concatenate([conv3, x], axis=3)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 240
    x = keras.layers.concatenate([conv2, x], axis=3)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 480
    x = keras.layers.concatenate([conv1, x], axis=3)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(output_channel_count, (1, 1), padding='same', activation=None)(x)

    model = keras.Model(inputs=model_input, outputs=x)
    return model


# My bigger Unet with halved l2
def get_model_10(input_shape, output_channel_count):
    model_input = keras.Input(shape=input_shape)

    conv1 = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(model_input)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(conv1)
    conv1 = keras.layers.BatchNormalization()(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)              # 240
    conv2 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(pool1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(conv2)
    conv2 = keras.layers.BatchNormalization()(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)              # 120
    conv3 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(pool2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv3)
    conv3 = keras.layers.BatchNormalization()(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)              # 60
    conv4 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(pool3)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv4)
    conv4 = keras.layers.BatchNormalization()(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)              # 30
    conv5 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(pool4)
    conv5 = keras.layers.BatchNormalization()(conv5)
    conv5 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(conv5)
    conv5 = keras.layers.BatchNormalization()(conv5)
    pool5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv5)              # 15

    x = keras.layers.UpSampling2D(interpolation='bilinear')(pool5)      # 30
    x = keras.layers.concatenate([conv5, x], axis=3)
    x = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 60
    x = keras.layers.concatenate([conv4, x], axis=3)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 120
    x = keras.layers.concatenate([conv3, x], axis=3)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 240
    x = keras.layers.concatenate([conv2, x], axis=3)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 480
    x = keras.layers.concatenate([conv1, x], axis=3)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(output_channel_count, (1, 1), padding='same', activation=None)(x)

    model = keras.Model(inputs=model_input, outputs=x)
    return model


# My bigger Unet with l2 on bias too
def get_model_11(input_shape, output_channel_count):
    model_input = keras.Input(shape=input_shape)

    conv1 = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(model_input)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(conv1)
    conv1 = keras.layers.BatchNormalization()(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)              # 240
    conv2 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(pool1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(conv2)
    conv2 = keras.layers.BatchNormalization()(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)              # 120
    conv3 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(pool2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(conv3)
    conv3 = keras.layers.BatchNormalization()(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)              # 60
    conv4 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(pool3)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(conv4)
    conv4 = keras.layers.BatchNormalization()(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)              # 30
    conv5 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(pool4)
    conv5 = keras.layers.BatchNormalization()(conv5)
    conv5 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(conv5)
    conv5 = keras.layers.BatchNormalization()(conv5)
    pool5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv5)              # 15

    x = keras.layers.UpSampling2D(interpolation='bilinear')(pool5)      # 30
    x = keras.layers.concatenate([conv5, x], axis=3)
    x = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 60
    x = keras.layers.concatenate([conv4, x], axis=3)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 120
    x = keras.layers.concatenate([conv3, x], axis=3)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 240
    x = keras.layers.concatenate([conv2, x], axis=3)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)      # 480
    x = keras.layers.concatenate([conv1, x], axis=3)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(output_channel_count, (1, 1), padding='same', activation=None)(x)

    model = keras.Model(inputs=model_input, outputs=x)
    return model


def get_hourglass_block_with_l2(stack_input):
    conv1 = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(stack_input)
    conv1 = keras.layers.BatchNormalization()(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)  # 240
    conv2 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(pool1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)  # 120
    conv3 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(pool2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)  # 60
    conv4 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(pool3)
    conv4 = keras.layers.BatchNormalization()(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)  # 30

    x = keras.layers.UpSampling2D(interpolation='bilinear')(pool4)  # 60
    x = keras.layers.concatenate([conv4, x], axis=3)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)  # 120
    x = keras.layers.concatenate([conv3, x], axis=3)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)  # 240
    x = keras.layers.concatenate([conv2, x], axis=3)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D(interpolation='bilinear')(x)  # 480
    x = keras.layers.concatenate([conv1, x], axis=3)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(x)
    x = keras.layers.BatchNormalization()(x)
    return x


def get_hourglass_model_with_l2(input_shape, output_channel_count, stack_size=2):
    model_input = keras.Input(shape=input_shape)
    prev_res = get_hourglass_block(model_input)
    x = None
    for index in range(stack_size - 1):
        if x is not None:
            prev_res = keras.layers.concatenate([prev_res, x], axis=3)
        x = get_hourglass_block(prev_res)

    output = keras.layers.Conv2D(output_channel_count, (1, 1), padding='same', activation=None)(x)
    model = keras.Model(inputs=model_input, outputs=output)
    return model


def get_model(input_shape, output_channel_count, model_number=0):
    if model_number == 0:
        return get_model_0(input_shape, output_channel_count)                             # without copies
    if model_number == 1:
        return get_model_1(input_shape, output_channel_count)                             # big UNet
    elif model_number == 2:
        return get_model_2(input_shape, output_channel_count)                             # lighter UNet
    elif model_number == 3:
        return get_hourglass_model(input_shape, output_channel_count, stack_size=2)
    elif model_number == 4:
        return get_hourglass_model(input_shape, output_channel_count, stack_size=3)
    elif model_number == 5:
        return get_hourglass_model(input_shape, output_channel_count, stack_size=4)
    elif model_number == 6:
        return get_hourglass_model(input_shape, output_channel_count, stack_size=5)
    elif model_number == 7:
        return unet(input_shape, output_channel_count)                                     # UNet from Inet
    elif model_number == 8:
        return get_model_8(input_shape, output_channel_count)                              # My bigger UNet
    elif model_number == 9:
        return get_model_9(input_shape, output_channel_count)                              # My bigger UNet with l2
    elif model_number == 10:
        return get_model_10(input_shape, output_channel_count)                              # My bigger UNet with halved l2
    elif model_number == 11:
        return get_model_11(input_shape, output_channel_count)                              # My bigger UNet with l2 on bias too
    elif model_number == 12:
        return get_hourglass_model_with_l2(input_shape, output_channel_count, stack_size=2) # Hourglass with l2
    elif model_number == 13:
        return get_hourglass_model_with_l2(input_shape, output_channel_count, stack_size=3) # Hourglass with l2
    elif model_number == 14:
        return get_hourglass_model_with_l2(input_shape, output_channel_count, stack_size=5) # Hourglass with l2
    return None

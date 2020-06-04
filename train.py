import argparse
import datetime
import numpy as np
import os
import tensorflow as tf

from models import get_model
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from validate_model import validate_model


def generate_augmented_data(X_train, Y_train, batch_size, seed=42):
    data_gen_args = dict(
        rotation_range=45.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode='constant',
        cval=0,
        validation_split=0.3,
    )

    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed, subset='training')
    Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed, subset='training')

    X_val_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed, subset='validation')
    Y_val_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed, subset='validation')

    train_generator = zip(X_train_augmented, Y_train_augmented)
    val_generator = zip(X_val_augmented, Y_val_augmented)
    return train_generator, val_generator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_number', type=int, default=0, help='model number')
    parser.add_argument('--epochs', type=int, default=100, help='epoch count')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--patience', type=int, default=50, help='epochs with no improvement until stop')
    parser.add_argument('--augment_data', action='store_true', help='whether to augment data')
    parser.add_argument('--validate_model', action='store_true', help='whether to validate model')
    parser.add_argument('--scale_factor', type=int, default=100000, help='scale factor for model validation')
    args = parser.parse_args()

    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

    print('loading data...')
    input_data_path = 'input'
    X_train = np.load(os.path.join(input_data_path, 'X_train.npy'))
    Y_train = np.load(os.path.join(input_data_path, 'Y_train.npy'))

    print('initializing model...')
    checkpoint_filename = os.path.join(input_data_path, 'best_model.hdf5')
    if os.path.exists(checkpoint_filename):
        print('loading weights...')
        model = keras.models.load_model(checkpoint_filename)
    else:
        model = get_model(input_shape=X_train[0].shape, output_channel_count=Y_train[0].shape[2],
                          model_number=args.model_number)
        model.compile(optimizer='Adam', loss='mse')

    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join('checkpoints', 'best_model.hdf5'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    csv_logger = keras.callbacks.CSVLogger(
        os.path.join('logs', 'train_log.csv'),
        append=True,
        separator=';'
    )
    log_dir = os.path.join('logs', 'fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopper = keras.callbacks.EarlyStopping(patience=args.patience)
    callbacks = [checkpoint, csv_logger, tensorboard_callback, early_stopper]

    if args.augment_data:
        print('augmentation started...')
        train_generator, val_generator = generate_augmented_data(X_train, Y_train, args.batch_size)

        print('train started...')
        model.fit_generator(
            train_generator,
            validation_data=val_generator,
            steps_per_epoch=len(X_train) * 0.7 / args.batch_size,
            validation_steps=len(X_train) * 0.3 / args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks
        )
    else:
        model.fit(
            x=X_train,
            y=Y_train,
            validation_split=0.3,
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks,
        )
    print('train ended')

    if args.validate_model:
        print('model validation started')
        print('last model')
        validate_model(model, X_train, Y_train, input_data_path, args.scale_factor)
        model.save(os.path.join('checkpoints', 'last_model.hdf5'))
        print()
        model.load(os.path.join('checkpoints', 'best_model.hdf5'))
        print('best model')
        validate_model(model, X_train, Y_train, input_data_path, args.scale_factor)

USING_TF2 = False

if USING_TF2:
    ### For Tensorflow 2.x
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import models
    from tensorflow.keras import layers
    from tensorflow.keras import optimizers
    from tensorflow.keras import preprocessing
    from tensorflow.keras import applications
else:
    ### For Keras
    import keras
    from keras import models
    from keras import layers
    from keras import optimizers
    from keras import preprocessing
    from keras import applications
import numpy as np


def build_model(clazz, input_shape: tuple, lr: float, epochs: int):
    ### Set Random Seed
    if USING_TF2:
        tf.random.set_seed(1)
    np.random.seed(1)

    ### Create and Configure Pre-trained Network
    conv_base = clazz(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
    )
    conv_base.trainable = False

    ### Create Model
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(
        optimizers.RMSprop(lr=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    # model.summary()
    return model


def build_generator(target_subdir: str, target_size: tuple, batch_size: int=20, shuffle: bool=True, **kwargs):
    import os
    base_dir = os.path.join(os.getcwd(), 'Datasets', 'cat_dog_car_bike')
    target_dir = os.path.join(base_dir, target_subdir)
    datagen = preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        **kwargs,
    )
    generator = datagen.flow_from_directory(
        target_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=shuffle,
    )
    return generator


def train_model(model, train_generator, val_generator, epochs: int):
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
    )
    return history


def main_for_train(clazz, classname: str, target_size: tuple, input_shape: tuple, lr: float, epochs: int):
    ### Filenames
    model_file = '{}_model_{:.0e}_{}.h5'.format(classname, lr, epochs)
    history_file = '{}_history_{:.0e}_{}.csv'.format(classname, lr, epochs)

    ### Build Model
    model = build_model(clazz, input_shape, lr, epochs)

    ### Build Generators
    train_generator = build_generator(
        'train',
        target_size,
        batch_size=20,
        shuffle=True,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )
    val_generator = build_generator(
        'val',
        target_size,
        batch_size=20,
        shuffle=True,
    )

    ### Train Model
    history = train_model(model, train_generator, val_generator, epochs)

    ### Save Model
    if model_file is not None:
        model.save(model_file)

    ### Save History
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    import csv
    with open(history_file, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(zip(range(1, epochs+1), acc, val_acc, loss, val_loss))


def main_for_predict(model_file: str, classname: str, target_size: tuple, lr: float, epochs: int):
    ### Filenames
    prediction_file = '{}_prediction_{:.0e}_{}.json'.format(classname, lr, epochs)

    ### Load Model
    model = models.load_model(model_file)

    ### Build Generator
    test_generator = build_generator(
        'test',
        target_size,
        batch_size=20,
        shuffle=False,
    )

    ### Make Predictions
    output_weights = model.predict(test_generator)

    ### Convert predicted results to dictionary
    from pathlib import Path
    import numpy as np
    categories = np.array(['cat', 'dog', 'car', 'motorbike'])
    predictions = categories[output_weights.argmax(axis=1)]
    filenames = test_generator.filenames
    stems = map(lambda x: Path(x).stem, filenames)
    predictionDict = dict(zip(stems, predictions))

    ### Saved resutls to q5_result/prediction.json
    import json
    with open(prediction_file, 'w', encoding='utf-8') as f:
        json.dump(predictionDict, f, indent=2, sort_keys=True)


def main():
    import sys

    ### Hyper-Parameters
    clazz = applications.ResNet50
    classname = 'ResNet50'
    target_size = (224, 224)
    input_shape = target_size + (3,)
    lr = 1e-4
    epochs = 100

    if len(sys.argv) < 2:
        print('Insufficient number of parameters')
        return

    if sys.argv[1] == 'train':
        lrs = [
            1e-4, 5e-4, 9e-4, 3e-4, 7e-4, 2e-4, 4e-4, 6e-4, 8e-4,
        ]
        for lr in lrs:
            print('--------')
            print(lr)
            print('--------')
            main_for_train(clazz, classname, target_size, input_shape, lr, epochs)
    elif sys.argv[1] == 'predict':
        if len(sys.argv) < 3:
            print('Missing model file argument')
            return
        model_file = sys.argv[2]
        main_for_predict(model_file, classname, target_size, lr, epochs)
    else:
        print('Invalid command: {}'.format(sys.argv[1]))


if __name__ == '__main__':
    ### Print Platform Information
    if USING_TF2:
        print('Tensorflow Version: {}'.format(tf.__version__))
        print('Physical Devices:')
        tf.config.list_physical_devices()
        print('Visible Devices:')
        tf.config.get_visible_devices()
    print('Keras Version: {}'.format(keras.__version__))
    print('Keras Backend: {}'.format(keras.backend.backend()))
    print('NumPy Version: {}'.format(np.__version__))
    main()

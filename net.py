import argparse
import shutil
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from matplotlib import pyplot as plt
import color_correct as cc


def adjust_image(img):
    return cc.max_white(img)


# Taken from https://github.com/keras-team/keras/issues/5862
def split_dataset(data_dir, training_data_dir, validata_data_dir, testing_data_dir, validation_split, test_split):
    # Recreate testing and training directories
    if testing_data_dir.count('/') > 1:
        shutil.rmtree(testing_data_dir, ignore_errors=False)
        os.makedirs(testing_data_dir)
        print("Successfully cleaned directory " + testing_data_dir)

    if training_data_dir.count('/') > 1:
        shutil.rmtree(training_data_dir, ignore_errors=False)
        os.makedirs(training_data_dir)
        print("Successfully cleaned directory " + training_data_dir)
    

    if validation_data_dir.count('/') > 1:
        shutil.rmtree(validation_data_dir, ignore_errors=False)
        os.makedirs(validation_data_dir)
        print("Successfully cleaned directory " + validation_data_dir)
   

    num_training_files = 0
    num_testing_files = 0
    num_validation_files = 0

    for subdir, dirs, files in os.walk(data_dir):
        category_name = os.path.basename(subdir)

        # Don't create a subdirectory for the root directory
        if category_name == os.path.basename(data_dir):
            continue

        training_data_category_dir = training_data_dir + '/' + category_name
        testing_data_category_dir = testing_data_dir + '/' + category_name
        validation_data_category_dir = validation_data_dir + '/' + category_name

        if not os.path.exists(training_data_category_dir):
            os.mkdir(training_data_category_dir)

        if not os.path.exists(testing_data_category_dir):
            os.mkdir(testing_data_category_dir)

        if not os.path.exists(validation_data_category_dir):
            os.mkdir(validation_data_category_dir)

        for file in files:
            input_file = os.path.join(subdir, file)
            r = np.random.ranf(1)
            if  r < validation_split:
                shutil.copy(input_file, validation_data_dir + '/' + category_name + '/' + file)
                num_validation_files += 1
            elif  r >= validation_split and r < validation_split + test_split:
                shutil.copy(input_file, test_data_dir + '/' + category_name + '/' + file)
                num_testing_files += 1
            else:
                shutil.copy(input_file, training_data_dir + '/' + category_name + '/' + file)
                num_training_files += 1

    print("Processed " + str(num_training_files) + " training files.")
    print("Processed " + str(num_validation_files) + " testing files.")
    print("Processed " + str(num_testing_files) + " testing files.")

    return num_training_files, num_validation_files, num_testing_files


# Check https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' Algorithm')
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    batch_size = 16
    img_width, img_height = 64, 64

    train_data_dir = 'data/train'
    validation_data_dir = 'data/validation'
    test_data_dir = 'data/test'
    test_split = 0.13
    validation_split = 0.17

    train_samples, validation_samples, test_samples = split_dataset('data/original', train_data_dir, validation_data_dir, test_data_dir, validation_split, test_split)


    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)


    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=adjust_image,
        fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1. / 255,
        preprocessing_function=adjust_image)

    test_datagen = ImageDataGenerator(rescale=1. / 255,
        preprocessing_function=adjust_image)


    train_generator = train_datagen.flow_from_directory(
        train_data_dir,  # this is the target directory
        target_size=(img_width, img_height),
      #  save_to_dir='data/generated/train',
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
      #  save_to_dir='data/generated/validation',
        batch_size=batch_size,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    epochs = 32

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation(K.relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation(K.relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation(K.relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # this converts our 3D feature maps to 1D feature vectors
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation(K.relu))
    model.add(Dropout(0.5))
    model.add(Dense(6))
    model.add(Activation(K.softmax))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size,
        workers=8)

    # always save your weights after training
    model.save_weights('first_try.h5')

    model.evaluate_generator(
        test_generator, 
        steps = test_samples // batch_size,
        workers = 8)

    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)

    plt.show()
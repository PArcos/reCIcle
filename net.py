from preprocess import *

nb_train_samples = 2527
nb_validation_samples = 2527
epochs = 2

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

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation(K.relu))
model.add(Dropout(0.5))
model.add(Dense(6))
model.add(Activation(K.softmax))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit_generator(
        train_generator,
        steps_per_epoch = nb_train_samples // batch_size,
        epochs = epochs,
        validation_data = validation_generator,
        validation_steps = nb_validation_samples // batch_size)
model.save_weights('first_try.h5')  # always save your weights after training or during training